from typing import List, Dict, Any, Tuple
import re
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from init.logger import logger
from base.llm import LLMManager
from prompt.entity_prompt import DIALOGUE_EXTRACTION_PROMPT, LOCOMO_EXTRACTION_PROMPT

class DialogueExtractor:
    def __init__(self, llm_manager: LLMManager, data_type: str = "LongmemEval"):
        self.llm = llm_manager
        self.data_type = data_type
        if data_type == "LOCOMO":
            self.extraction_prompt = LOCOMO_EXTRACTION_PROMPT
            logger.info("Dialogue Extractor initialized with LOCOMO prompt")
        else:
            self.extraction_prompt = DIALOGUE_EXTRACTION_PROMPT
            logger.info("Dialogue Extractor initialized with LongmemEval prompt")

    async def extract_entities_and_relationships(self, text: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Extract both entities and relationships from dialogue text using the combined prompt."""
        if not text:
            return [], []

        try:
            mock_json = {
                "session_time": "unknown",
                "text": text, 
                "session_id": "unknown"
            }
            
            formatted_text = json.dumps(mock_json)
            logger.debug(f"Converted plain text to JSON format for prompt: {formatted_text[:100]}")
            prompt = self.extraction_prompt.format(text=formatted_text)
            response = await self.llm.generate(prompt)
            
            entities, relationships = self._parse_dialogue_response(response)
            logger.debug(f"Extracted {len(entities)} entities and {len(relationships)} relationships from dialogue")
            return entities, relationships
            
        except Exception as e:
            logger.error(f"Failed to extract entities and relationships from dialogue: {e}")
            logger.error(f"Input text that caused error: {text[:200]}")
            return [], []

    def _parse_dialogue_response(self, response: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Parse the LLM response to extract entities and relationships."""
        entities = []
        relationships = []
        
        if not response:
            return entities, relationships
        parts = response.split('##')
        
        for part in parts:
            part = part.strip()
            if not part or part == 'END':
                continue
                
            if part.startswith('("entity"|'):
                entity = self._parse_entity(part)
                if entity:
                    entities.append(entity)
            elif part.startswith('("relationship"|'):
                relationship = self._parse_relationship(part)
                if relationship:
                    relationships.append(relationship)
        
        return entities, relationships

    def _parse_entity(self, entity_str: str) -> Dict[str, Any]:
        """Parse entity string format: ("entity"|<entity_name>|<entity_type>)"""
        try:
            # Remove the outer parentheses and split by |
            content = entity_str.strip('()')
            parts = [part.strip('"') for part in content.split('|')]
            
            if len(parts) >= 3 and parts[0] == 'entity':
                return {
                    'entity': parts[1],
                    'type': parts[2],
                    'description': parts[3] if len(parts) > 3 else ''
                }
        except Exception as e:
            logger.warning(f"Failed to parse entity: {entity_str}, error: {e}")
        
        return None

    def _parse_relationship(self, relationship_str: str) -> Dict[str, Any]:
        """Parse relationship string format: ("relationship"|<create_time>|<session_id>|<source_entity>|<target_entity>|<relationship_name>|<relationship_strength>)"""
        try:
            # Remove the outer parentheses and split by |
            content = relationship_str.strip('()')
            parts = [part.strip('"') for part in content.split('|')]
            
            if len(parts) >= 7 and parts[0] == 'relationship':
                return {
                    'create_time': parts[1],
                    'session_id': parts[2],  # Fixed: was 'origin', now 'session_id' to match prompt
                    'src': parts[3],
                    'tgt': parts[4],
                    'relation': parts[5],
                    'strength': int(parts[6]) if parts[6].isdigit() else 1,
                    'weight': float(parts[6]) / 10.0 if parts[6].isdigit() else 0.1  # Convert strength to weight
                }
        except Exception as e:
            logger.warning(f"Failed to parse relationship: {relationship_str}, error: {e}")
        
        return None

    async def extract_from_chunks(self, chunks: List[Dict[str, Any]], progress_bar=None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Extract entities and relationships from multiple dialogue chunks with concurrent support.
        
        Args:
            chunks: List of chunks to process
            progress_bar: Optional tqdm progress bar to update as each request completes
        """
        if not chunks:
            return [], []
        
        if hasattr(self.llm, 'enable_concurrent') and self.llm.enable_concurrent:
            prompts = []
            chunk_metas = []
            
            for chunk in chunks:
                text = chunk.get('text', '')
                
                # Get session metadata from chunk for better context
                session_time = chunk.get('session_time', 'unknown')
                session_id = chunk.get('session_id', 'unknown')
                
                # Create JSON structure with session metadata (matching prompt format)
                enhanced_json = {
                    "text": text,
                    "session_time": str(session_time),
                    "session_id": str(session_id)
                }
                
                # Use the enhanced JSON format
                formatted_text = json.dumps(enhanced_json)
                prompt = self.extraction_prompt.format(text=formatted_text)
                prompts.append(prompt)
                chunk_metas.append(chunk)
            
            try:
                # Pass progress_bar to batch_generate so it updates as each request completes
                responses = await self.llm.batch_generate(prompts, progress_bar=progress_bar)
            except Exception as e:
                logger.error(f"Failed to batch extract from chunks: {e}")
                responses = ["" for _ in prompts]
                # Update progress bar for failed requests
                if progress_bar:
                    progress_bar.update(len(prompts))
            
            all_entities = []
            all_relationships = []
            
            for i, (response, chunk) in enumerate(zip(responses, chunk_metas)):
                try:
                    chunk_entities, chunk_relationships = self._parse_dialogue_response(response)
                    
                    # Add chunk metadata to entities
                    for entity in chunk_entities:
                        entity['chunk_id'] = chunk.get('chunk_id', 0) 
                        entity['source_text'] = chunk.get('text', '')[:100] + '...' if len(chunk.get('text', '')) > 100 else chunk.get('text', '')
                    
                    # Add chunk metadata to relationships
                    for relationship in chunk_relationships:
                        relationship['chunk_id'] = chunk.get('chunk_id', 0)
                    
                    all_entities.extend(chunk_entities)
                    all_relationships.extend(chunk_relationships)
                except Exception as e:
                    logger.error(f"Failed to parse response for chunk {i}: {e}")
                    continue
        else:
            all_entities = []
            all_relationships = []

            for chunk in chunks:
                text = chunk.get('text', '')
                
                session_time = chunk.get('session_time', 'unknown')
                session_id = chunk.get('session_id', 'unknown')
                enhanced_json = {
                    "text": text,
                    "session_time": str(session_time),
                    "session_id": str(session_id)
                }
                
                formatted_text = json.dumps(enhanced_json)
                try:
                    prompt = self.extraction_prompt.format(text=formatted_text)
                    response = await self.llm.generate(prompt)
                    chunk_entities, chunk_relationships = self._parse_dialogue_response(response)
                except Exception as e:
                    logger.error(f"Failed to extract from chunk: {e}")
                    chunk_entities, chunk_relationships = [], []
                finally:
                    if progress_bar:
                        progress_bar.update(1)

                # Add chunk metadata to entities
                for entity in chunk_entities:
                    entity['chunk_id'] = chunk.get('chunk_id', chunk.get('doc_id', 0))
                    entity['source_text'] = text[:100] + '...' if len(text) > 100 else text
                    entity['session_time'] = session_time
                    entity['session_id'] = session_id

                # Add chunk metadata to relationships
                for relationship in chunk_relationships:
                    relationship['chunk_id'] = chunk.get('chunk_id', chunk.get('doc_id', 0))
                    relationship['source_text'] = text[:100] + '...' if len(text) > 100 else text
                    # Preserve create_time from LLM extraction, fallback to session_time from chunk
                    if 'create_time' not in relationship or not relationship['create_time']:
                        relationship['create_time'] = session_time
                    # Also set session_time for compatibility
                    if 'session_time' not in relationship or not relationship['session_time']:
                        relationship['session_time'] = relationship.get('create_time', session_time)
                    # Set session_id if not already present
                    if 'session_id' not in relationship or not relationship['session_id']:
                        relationship['session_id'] = session_id

                all_entities.extend(chunk_entities)
                all_relationships.extend(chunk_relationships)

        logger.info(f"Extracted {len(all_entities)} entities and {len(all_relationships)} relationships from {len(chunks)} dialogue chunks")
        return all_entities, all_relationships

    def deduplicate_entities(self, entities: List[Dict[str, Any]], 
                           similarity_threshold: float = 0.85) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
        if not entities:
            return [], {}

        unique_entities = []
        entity_mapping = {}

        for entity in entities:
            entity_name = entity.get('entity', '')
            entity_name_lower = entity_name.lower()
            entity_type = entity.get('type', '').lower()
            is_duplicate = False
            canonical_entity = None

            for unique_entity in unique_entities:
                unique_name = unique_entity.get('entity', '')
                unique_name_lower = unique_name.lower()
                unique_type = unique_entity.get('type', '').lower()
                if entity_name_lower == unique_name_lower and entity_type == unique_type:
                    is_duplicate = True
                    canonical_entity = unique_entity
                    self._merge_entity_sessions(unique_entity, entity)
                    break
                
                similarity = self._calculate_similarity(entity_name_lower, unique_name_lower)
                if similarity >= similarity_threshold and self._are_types_compatible(entity_type, unique_type):
                    is_duplicate = True
                    canonical_entity = unique_entity
                    self._merge_entity_sessions(unique_entity, entity)
                    break

            if is_duplicate:
                # Record the mapping from duplicate to canonical name
                canonical_name = canonical_entity.get('entity', '')
                entity_mapping[entity_name] = canonical_name  # Map original names
                logger.debug(f"Entity mapping: '{entity_name}' -> '{canonical_name}'")
            else:
                unique_entities.append(entity)
                # Entity is its own canonical name
                entity_mapping[entity_name] = entity_name

        return unique_entities, entity_mapping

    def deduplicate_relationships(self, relationships: List[Dict[str, Any]], 
                                entity_mapping: Dict[str, str] = None,
                                similarity_threshold: float = 0.9) -> List[Dict[str, Any]]:

        if not relationships:
            return []
        if entity_mapping:
            relationships = self._update_relationship_entity_references(relationships, entity_mapping)

        unique_relationships = []

        for relationship in relationships:
            src = relationship.get('src', '').lower()
            tgt = relationship.get('tgt', '').lower()
            relation = relationship.get('relation', '').lower()
            is_duplicate = False

            for unique_relationship in unique_relationships:
                unique_src = unique_relationship.get('src', '').lower()
                unique_tgt = unique_relationship.get('tgt', '').lower()
                unique_relation = unique_relationship.get('relation', '').lower()
                
                # Exact match
                if src == unique_src and tgt == unique_tgt and relation == unique_relation:
                    is_duplicate = True
                    # Merge session information and update strength
                    self._merge_relationship_sessions(unique_relationship, relationship)
                    break

            if not is_duplicate:
                unique_relationships.append(relationship)

        return unique_relationships
    
    def _update_relationship_entity_references(self, relationships: List[Dict[str, Any]], 
                                             entity_mapping: Dict[str, str]) -> List[Dict[str, Any]]:
        updated_relationships = []
        updates_count = 0
        
        for relationship in relationships:
            updated_relationship = relationship.copy()
            
            # Update src entity reference
            src = relationship.get('src', '')
            if src in entity_mapping and entity_mapping[src] != src:
                updated_relationship['src'] = entity_mapping[src]
                updates_count += 1
                logger.debug(f"Updated src: '{src}' -> '{entity_mapping[src]}'")
                
            # Update tgt entity reference  
            tgt = relationship.get('tgt', '')
            if tgt in entity_mapping and entity_mapping[tgt] != tgt:
                updated_relationship['tgt'] = entity_mapping[tgt]
                updates_count += 1
                logger.debug(f"Updated tgt: '{tgt}' -> '{entity_mapping[tgt]}'")
                
            updated_relationships.append(updated_relationship)
            
        return updated_relationships

    def _merge_entity_sessions(self, existing_entity: Dict[str, Any], new_entity: Dict[str, Any]):
        if 'session_time' in new_entity:
            existing_sessions = existing_entity.get('session_times', [])
            if 'session_time' in existing_entity:
                existing_sessions.append(existing_entity['session_time'])
                existing_entity['session_times'] = list(set(existing_sessions + [new_entity['session_time']]))
            else:
                existing_entity['session_times'] = [new_entity['session_time']]
            existing_entity['session_time'] = new_entity['session_time']  # Keep latest
        
        if 'session_id' in new_entity:
            existing_sessions = existing_entity.get('session_ids', [])
            if 'session_id' in existing_entity:
                existing_sessions.append(existing_entity['session_id'])
                existing_entity['session_ids'] = list(set(existing_sessions + [new_entity['session_id']]))
            else:
                existing_entity['session_ids'] = [new_entity['session_id']]
            existing_entity['session_id'] = new_entity['session_id']  # Keep latest
        
        if 'chunk_id' in new_entity:
            existing_chunks = existing_entity.get('chunk_ids', [])
            if 'chunk_id' in existing_entity:
                existing_chunks.append(existing_entity['chunk_id'])
                existing_entity['chunk_ids'] = list(set(existing_chunks + [new_entity['chunk_id']]))
            else:
                existing_entity['chunk_ids'] = [new_entity['chunk_id']]
            existing_entity['chunk_id'] = new_entity['chunk_id']  # Keep latest

    def _merge_relationship_sessions(self, existing_relationship: Dict[str, Any], new_relationship: Dict[str, Any]):
        """Merge session information for duplicate relationships and update strength."""
        # Update strength (take the higher value)
        existing_strength = existing_relationship.get('strength', 1)
        new_strength = new_relationship.get('strength', 1)
        existing_relationship['strength'] = max(existing_strength, new_strength)
        existing_relationship['weight'] = existing_relationship['strength'] / 10.0
        
        if 'session_time' in new_relationship and new_relationship['session_time']:
            existing_sessions = existing_relationship.get('session_times', [])
            if 'session_time' in existing_relationship and existing_relationship['session_time']:
                existing_sessions.append(existing_relationship['session_time'])
                existing_relationship['session_times'] = list(set(existing_sessions + [new_relationship['session_time']]))
            else:
                existing_relationship['session_times'] = [new_relationship['session_time']]
            existing_relationship['session_time'] = new_relationship['session_time']
        
        if 'session_id' in new_relationship:
            existing_sessions = existing_relationship.get('session_ids', [])
            if 'session_id' in existing_relationship:
                existing_sessions.append(existing_relationship['session_id'])
                existing_relationship['session_ids'] = list(set(existing_sessions + [new_relationship['session_id']]))
            else:
                existing_relationship['session_ids'] = [new_relationship['session_id']]
            existing_relationship['session_id'] = new_relationship['session_id']
        
        # Merge chunk IDs
        if 'chunk_id' in new_relationship:
            existing_chunks = existing_relationship.get('chunk_ids', [])
            if 'chunk_id' in existing_relationship:
                existing_chunks.append(existing_relationship['chunk_id'])
                existing_relationship['chunk_ids'] = list(set(existing_chunks + [new_relationship['chunk_id']]))
            else:
                existing_relationship['chunk_ids'] = [new_relationship['chunk_id']]
            existing_relationship['chunk_id'] = new_relationship['chunk_id']  # Keep latest

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple similarity between two texts."""
        if not text1 or not text2:
            return 0.0
        set1 = set(text1.split())
        set2 = set(text2.split())
        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union > 0 else 0.0

    def _are_types_compatible(self, type1: str, type2: str) -> bool:
        """Check if two entity types are compatible for merging."""
        return type1 == type2
