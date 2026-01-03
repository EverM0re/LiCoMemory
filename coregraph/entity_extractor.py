from typing import List, Dict, Any
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from init.logger import logger
from base.llm import LLMManager

class EntityExtractor:
    """Extractor for named entities from text."""

    def __init__(self, llm_manager: LLMManager):
        """Initialize entity extractor."""
        self.llm = llm_manager
        logger.info("Entity Extractor initialized")

    async def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities from text."""
        if not text:
            return []
        try:
            entities = await self.llm.extract_entities(text)
            logger.debug(f"Extracted {len(entities)} entities from text")
            return entities
        except Exception as e:
            logger.error(f"Failed to extract entities: {e}")
            return []

    async def extract_from_chunks(self, chunks: List[Dict[str, Any]], progress_bar=None) -> List[Dict[str, Any]]:
        """Extract entities from multiple chunks with concurrent support.
        
        Args:
            chunks: List of chunks to process
            progress_bar: Optional tqdm progress bar to update as each request completes
        """
        if not chunks:
            return []
        
        if hasattr(self.llm, 'enable_concurrent') and self.llm.enable_concurrent:
            logger.info(f"Extracting entities from {len(chunks)} chunks concurrently")
            texts = [chunk.get('text', '') for chunk in chunks]
            try:
                # Pass progress_bar to batch_extract_entities so it updates as each request completes
                chunk_entities_list = await self.llm.batch_extract_entities(texts, progress_bar=progress_bar)
            except Exception as e:
                logger.error(f"Failed to extract entities from chunks: {e}")
                chunk_entities_list = [[] for _ in chunks]
                # Update progress bar for failed requests
                if progress_bar:
                    progress_bar.update(len(chunks))
            
            all_entities = []
            for i, (chunk, chunk_entities) in enumerate(zip(chunks, chunk_entities_list)):
                for entity in chunk_entities:
                    entity['chunk_id'] = chunk.get('chunk_id', 0)
                    entity['source_text'] = chunk.get('text', '')[:100] + '...' if len(chunk.get('text', '')) > 100 else chunk.get('text', '')
                all_entities.extend(chunk_entities)
        else:
            all_entities = []
            for chunk in chunks:
                text = chunk.get('text', '')
                try:
                    chunk_entities = await self.extract_entities(text)
                except Exception as e:
                    logger.error(f"Failed to extract entities from chunk {chunk.get('chunk_id')}: {e}")
                    chunk_entities = []
                finally:
                    if progress_bar:
                        progress_bar.update(1)

                for entity in chunk_entities:
                    entity['chunk_id'] = chunk.get('chunk_id', 0)
                    entity['source_text'] = text[:100] + '...' if len(text) > 100 else text

                all_entities.extend(chunk_entities)

        logger.info(f"Extracted {len(all_entities)} entities from {len(chunks)} chunks")
        return all_entities

    def deduplicate_entities(self, entities: List[Dict[str, Any]],
                           similarity_threshold: float = 0.85) -> List[Dict[str, Any]]:
        """Deduplicate entities based on similarity."""
        if not entities:
            return []

        unique_entities = []

        for entity in entities:
            entity_name = entity.get('entity', '').lower()
            is_duplicate = False

            for unique_entity in unique_entities:
                unique_name = unique_entity.get('entity', '').lower()
                similarity = self._calculate_similarity(entity_name, unique_name)

                if similarity >= similarity_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_entities.append(entity)

        return unique_entities

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        if not text1 or not text2:
            return 0.0

        set1 = set(text1.split())
        set2 = set(text2.split())

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union > 0 else 0.0
