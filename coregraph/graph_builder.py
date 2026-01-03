from typing import List, Dict, Any
import networkx as nx
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from init.logger import logger
from init.config import Config

class GraphBuilder:
    """Builder for constructing knowledge graphs."""

    def __init__(self, config: Config):
        """Initialize graph builder."""
        self.config = config
        self.graph = nx.DiGraph()
        logger.info("Graph Builder initialized")

    def add_entity(self, entity: Dict[str, Any]) -> None:
        """Add entity to graph."""
        entity_name = entity.get('entity_name', '')
        if entity_name:
            self.graph.add_node(entity_name, **entity)
            logger.debug(f"Added entity: {entity_name}")

    def add_relationship(self, relationship: Dict[str, Any]) -> None:
        """Add relationship to graph."""
        src = relationship.get('src_id', '')
        tgt = relationship.get('tgt_id', '')
        relation = relationship.get('relation_name', '')

        if src and tgt and relation:
            self.graph.add_edge(src, tgt, **relationship)
            logger.debug(f"Added relationship: {src} -> {relation} -> {tgt}")

    def build_from_entities_and_relationships(self, entities: List[Dict[str, Any]], 
                                             relationships: List[Dict[str, Any]]) -> None:
        logger.debug(f"Adding {len(entities)} entities to graph")
        for entity in entities:
            entity_data = {
                'entity_name': entity.get('entity', ''),
                'entity_type': entity.get('type', 'unknown'),
                'chunk_id': entity.get('chunk_id', ''),
                'description': entity.get('description', '')
            }
            self.add_entity(entity_data)
        
        logger.debug(f"Adding {len(relationships)} relationships to graph")
        for relationship in relationships:
            # Prefer create_time from LLM extraction, fallback to session_time from chunk
            session_time = relationship.get('create_time', '') or relationship.get('session_time', '')
            
            relationship_data = {
                'src_id': relationship.get('src', ''),
                'tgt_id': relationship.get('tgt', ''),
                'relation_name': relationship.get('relation', ''),
                'chunk_id': relationship.get('chunk_id', ''),
                'session_id': relationship.get('session_id', ''),
                'session_time': session_time,  # Use create_time if available
                'description': relationship.get('description', '')
            }
            self.add_relationship(relationship_data)
        
        self._remove_isolated_nodes()

    def get_graph_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        return {
            'num_nodes': len(self.graph.nodes),
            'num_edges': len(self.graph.edges),
            'node_types': self._get_node_type_counts(),
            'edge_types': self._get_edge_type_counts()
        }

    def _get_node_type_counts(self) -> Dict[str, int]:
        """Get counts of different node types."""
        type_counts = {}
        for node, data in self.graph.nodes(data=True):
            node_type = data.get('entity_type', 'unknown')
            type_counts[node_type] = type_counts.get(node_type, 0) + 1
        return type_counts

    def _get_edge_type_counts(self) -> Dict[str, int]:
        """Get counts of different edge types."""
        type_counts = {}
        for _, _, data in self.graph.edges(data=True):
            edge_type = data.get('relation_name', 'unknown')
            type_counts[edge_type] = type_counts.get(edge_type, 0) + 1
        return type_counts
    
    def _remove_isolated_nodes(self) -> None:
        """Remove isolated nodes (nodes with no edges) from the graph."""
        isolated_nodes = [node for node in self.graph.nodes() if self.graph.degree(node) == 0]
        
        if isolated_nodes:
            self.graph.remove_nodes_from(isolated_nodes)
    
    def add_entities_and_relationships_incrementally(self, entities: List[Dict[str, Any]], 
                                                    relationships: List[Dict[str, Any]]) -> None:
        logger.info(f"🔄 Incrementally adding {len(entities)} entities and {len(relationships)} relationships")
        
        new_entities = 0
        merged_entities = 0
        new_relationships = 0
        
        for entity in entities:
            entity_name = entity.get('entity', '')
            if not entity_name:
                continue
                
            entity_data = {
                'entity_name': entity_name,
                'entity_type': entity.get('type', 'unknown'),
                'chunk_id': entity.get('chunk_id', ''),
                'description': entity.get('description', '')
            }
            
            if entity_name in self.graph.nodes:
                existing_data = self.graph.nodes[entity_name]
                merged_data = self._merge_entity_attributes(existing_data, entity_data)
                self.graph.nodes[entity_name].update(merged_data)
                merged_entities += 1
                logger.debug(f"Merged entity: {entity_name}")
            else:
                self.add_entity(entity_data)
                new_entities += 1
                logger.debug(f"Added new entity: {entity_name}")
        
        for relationship in relationships:
            relationship_data = {
                'src_id': relationship.get('src', ''),
                'tgt_id': relationship.get('tgt', ''),
                'relation_name': relationship.get('relation', ''),
                'chunk_id': relationship.get('chunk_id', ''),
                'session_id': relationship.get('session_id', ''),
                'session_time': relationship.get('session_time', ''),
                'description': relationship.get('description', '')
            }
            
            src = relationship_data['src_id']
            tgt = relationship_data['tgt_id']
            
            if src and tgt and relationship_data['relation_name']:
                if self.graph.has_edge(src, tgt):
                    existing_edge_data = self.graph.edges[src, tgt]
                    self._merge_relationship_data(existing_edge_data, relationship_data)
                    logger.debug(f"Merged relationship: {src} -> {relationship_data['relation_name']} -> {tgt}")
                else:
                    self.add_relationship(relationship_data)
                    new_relationships += 1
                    logger.debug(f"Added new relationship: {src} -> {relationship_data['relation_name']} -> {tgt}")
        
        logger.info(f"✅ Incremental addition completed:")
        logger.info(f"   - New entities: {new_entities}")
        logger.info(f"   - Merged entities: {merged_entities}")
        logger.info(f"   - New relationships: {new_relationships}")
        logger.info(f"   - Total graph: {len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges")
        
        self._remove_isolated_nodes()
        
        logger.info(f"After cleaning: {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges")
    
    def _merge_entity_attributes(self, existing: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
        """Merge attributes of existing and new entity data."""
        merged = existing.copy()
        
        for key, value in new.items():
            if value and (key not in merged or not merged[key]):
                merged[key] = value
            elif key == 'chunk_id' and value:
                existing_chunk = merged.get('chunk_id', '')
                if existing_chunk != value:
                    merged[key] = value
        
        return merged
    
    def _merge_relationship_data(self, existing: Dict[str, Any], new: Dict[str, Any]) -> None:
        """Merge chunk_id and session_id from new relationship into existing one."""
        existing_strength = existing.get('strength', 1)
        new_strength = new.get('strength', 1)
        existing['strength'] = max(existing_strength, new_strength)
        existing['weight'] = existing['strength'] / 10.0
        
        existing_chunks = existing.get('chunk_ids', [])
        
        if 'chunk_id' in existing and existing['chunk_id']:
            if existing['chunk_id'] not in existing_chunks:
                existing_chunks.append(existing['chunk_id'])
        
        if 'chunk_ids' in new and new['chunk_ids']:
            existing_chunks.extend(new['chunk_ids'])
        elif 'chunk_id' in new and new['chunk_id']:
            existing_chunks.append(new['chunk_id'])
        
        existing['chunk_ids'] = list(set(existing_chunks))
        if 'chunk_id' in new:
            existing['chunk_id'] = new['chunk_id']
        
        existing_sessions = existing.get('session_ids', [])
        
        if 'session_id' in existing and existing['session_id']:
            if existing['session_id'] not in existing_sessions:
                existing_sessions.append(existing['session_id'])
        
        if 'session_ids' in new and new['session_ids']:
            existing_sessions.extend(new['session_ids'])
        elif 'session_id' in new and new['session_id']:
            existing_sessions.append(new['session_id'])
        
        existing['session_ids'] = list(set(existing_sessions))
        if 'session_id' in new:
            existing['session_id'] = new['session_id']
        
        existing_times = existing.get('session_times', [])
        
        if 'session_time' in existing and existing['session_time']:
            if existing['session_time'] not in existing_times:
                existing_times.append(existing['session_time'])
        
        if 'session_times' in new and new['session_times']:
            existing_times.extend(new['session_times'])
        elif 'session_time' in new and new['session_time']:
            existing_times.append(new['session_time'])
        
        existing['session_times'] = list(set(existing_times))
        if 'session_time' in new and new['session_time']:
            existing['session_time'] = new['session_time']
        
        logger.debug(f"Merged relationship data: chunk_ids={existing.get('chunk_ids', [])}, session_ids={existing.get('session_ids', [])}, strength={existing.get('strength', 1)}")
