from typing import List, Dict, Any
import networkx as nx
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from init.logger import logger
from init.config import Config

class Retriever:

    def __init__(self, config: Config, graph: nx.DiGraph):
        self.config = config
        self.graph = graph
        logger.info("Retriever initialized")

    def retrieve_entities(self, entities: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        relevant_entities = []

        for entity in entities:
            if entity in self.graph:
                entity_data = dict(self.graph.nodes[entity])
                entity_data['entity_name'] = entity
                relevant_entities.append(entity_data)

        relevant_entities = relevant_entities[:top_k]

        logger.info(f"Retrieved {len(relevant_entities)} entities")
        return relevant_entities

    def retrieve_relationships(self, entities: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        relevant_relationships = []

        for entity in entities:
            if entity in self.graph:
                edges = []
                edges.extend(self.graph.in_edges(entity, data=True))
                edges.extend(self.graph.out_edges(entity, data=True))

                for src, tgt, data in edges:
                    relationship_data = dict(data)
                    relationship_data.update({
                        'src': src,
                        'tgt': tgt
                    })
                    relevant_relationships.append(relationship_data)

        seen = set()
        unique_relationships = []
        for rel in relevant_relationships:
            rel_key = (rel['src'], rel['tgt'], rel.get('relation_name', ''))
            if rel_key not in seen:
                seen.add(rel_key)
                unique_relationships.append(rel)

        unique_relationships = unique_relationships[:top_k]

        logger.info(f"Retrieved {len(unique_relationships)} relationships")
        return unique_relationships

    def retrieve_chunks(self, relationships: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        chunk_ids = set()

        for rel in relationships:
            chunk_id = rel.get('chunk_id')
            if chunk_id is not None:
                chunk_ids.add(chunk_id)
        chunks = [{'chunk_id': chunk_id} for chunk_id in chunk_ids]

        logger.info(f"Retrieved {len(chunks)} chunks")
        return chunks
