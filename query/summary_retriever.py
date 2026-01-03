import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Dict, Any, Tuple
import json
import numpy as np
from init.logger import logger
from base.embeddings import EmbeddingManager
from prompt.query_prompt import SUMMARY_QUERY_PROMPT
from base.llm import LLMManager


class SummaryRetriever:
    
    def __init__(self, embedding_manager: EmbeddingManager, llm_manager: LLMManager, config=None):
        self.embedding_manager = embedding_manager
        self.llm = llm_manager
        self.config = config
        self.summaries = []
        self.summary_embeddings = None
        logger.info("Summary Retriever initialized")
    
    def load_summaries(self, summary_path: str) -> None:
        try:
            with open(summary_path, 'r', encoding='utf-8') as f:
                self.summaries = json.load(f)
            logger.info(f"Loaded {len(self.summaries)} session summaries")
        except Exception as e:
            logger.error(f"Failed to load summaries from {summary_path}: {e}")
            self.summaries = []
    
    async def build_summary_embeddings(self) -> None:
        if not self.summaries:
            logger.warning("No summaries available for embedding")
            return
        
        logger.info(f"Building embeddings for {len(self.summaries)} session summaries")
        
        precomputed_embeddings = []
        texts_to_compute = []
        indices_to_compute = []
        
        for i, summary in enumerate(self.summaries):
            if 'embedding' in summary and summary['embedding']:
                precomputed_embeddings.append(summary['embedding'])
            else:
                precomputed_embeddings.append(None)
                summary_text = self._extract_summary_text(summary)
                texts_to_compute.append(summary_text)
                indices_to_compute.append(i)
        
        if texts_to_compute:
            logger.info(f"Computing embeddings for {len(texts_to_compute)} summaries without precomputed embeddings")
            computed_embeddings = await self.embedding_manager.get_embeddings(texts_to_compute)
            if computed_embeddings and len(computed_embeddings) == len(texts_to_compute):
                for idx, computed_emb in zip(indices_to_compute, computed_embeddings):
                    precomputed_embeddings[idx] = computed_emb
        else:
            logger.info(f"✅ Using precomputed embeddings for all {len(self.summaries)} summaries")
        
        self.summary_embeddings = precomputed_embeddings if None not in precomputed_embeddings else []
        
        if self.summary_embeddings:
            logger.info(f"Built embeddings for {len(self.summary_embeddings)} summaries")
    
    def _extract_summary_text(self, summary: Dict[str, Any]) -> str:
        text_parts = []
        session_id = summary.get('session_id', 'unknown')
        session_time = summary.get('session_time', 'unknown')
        text_parts.append(f"Session {session_id} ({session_time})")
        
        if 'keys' in summary and isinstance(summary['keys'], str):
            text_parts.append(f"Keys: {summary['keys']}")
        
        if 'context' in summary and isinstance(summary['context'], dict):
            context = summary['context']
            theme_keys = [k for k in context.keys() if k.startswith('theme_')]
            theme_keys.sort(key=lambda x: int(x.split('_')[1]) if x.split('_')[1].isdigit() else 0)
            
            for theme_key in theme_keys:
                theme_num = theme_key.split('_')[1]
                summary_key = f'summary_{theme_num}'
                
                theme_title = context.get(theme_key, '')
                theme_summary = context.get(summary_key, '')
                
                if theme_title:
                    text_parts.append(f"{theme_key}: {theme_title}")
                if theme_summary:
                    text_parts.append(f"Summary: {theme_summary}")
        
        return ' '.join(text_parts)
    
    async def retrieve_relevant_summaries(self, question: str, top_k: int = None) -> List[Dict[str, Any]]:
        if top_k is None and self.config:
            top_k = getattr(self.config.retriever, 'top_summary', 2)
        elif top_k is None:
            top_k = 2  # Default value if config is not available
        
        if not self.summaries or self.summary_embeddings is None:
            logger.warning("No summaries or embeddings available")
            return []
        
        logger.info(f"Retrieving top {top_k} relevant summaries for question using entity-based matching")
        
        entities = []
        if self.llm:
            entities_data = await self.llm.extract_entities(question)
            entities = [entity.get('entity', '') for entity in entities_data if entity.get('entity')]
            logger.info(f"Extracted entities from question: {entities}")
        else:
            logger.warning("LLMManager not available, cannot extract entities")
        
        if not entities:
            logger.warning("No entities extracted from question or LLM unavailable, falling back to direct question matching")
            question_embedding = await self.embedding_manager.get_embeddings([question])
            if not question_embedding:
                logger.error("Failed to get question embedding")
                return []
            question_embedding = question_embedding[0]
            similarities = []
            for i, summary_embedding in enumerate(self.summary_embeddings):
                similarity = self.embedding_manager.cosine_similarity(question_embedding, summary_embedding)
                similarities.append((i, similarity))
            
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_indices = [idx for idx, _ in similarities[:top_k]]
            relevant_summaries = [self.summaries[i] for i in top_indices]
            logger.info(f"Retrieved {len(relevant_summaries)} relevant summaries using direct question matching")
            return relevant_summaries
        
        entity_embeddings = await self.embedding_manager.get_embeddings(entities)
        if not entity_embeddings or len(entity_embeddings) != len(entities):
            logger.error("Failed to get embeddings for entities, falling back to direct question matching")
            question_embedding = await self.embedding_manager.get_embeddings([question])
            if not question_embedding:
                logger.error("Failed to get question embedding")
                return []
            question_embedding = question_embedding[0]
            
            similarities = []
            for i, summary_embedding in enumerate(self.summary_embeddings):
                similarity = self.embedding_manager.cosine_similarity(question_embedding, summary_embedding)
                similarities.append((i, similarity))
            
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_indices = [idx for idx, _ in similarities[:top_k]]
            relevant_summaries = [self.summaries[i] for i in top_indices]
            logger.info(f"Retrieved {len(relevant_summaries)} relevant summaries using direct question matching after entity embedding failure")
            return relevant_summaries
        
        # Calculate similarities between each entity and summaries, aggregate scores
        summary_scores = [0.0] * len(self.summaries)
        for entity_idx, entity_embedding in enumerate(entity_embeddings):
            for summary_idx, summary_embedding in enumerate(self.summary_embeddings):
                similarity = self.embedding_manager.cosine_similarity(entity_embedding, summary_embedding)
                summary_scores[summary_idx] = max(summary_scores[summary_idx], similarity)  # Use max similarity for each summary
        
        # Create list of (index, score) pairs
        similarities = [(i, score) for i, score in enumerate(summary_scores)]
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in similarities[:top_k]]
        
        relevant_summaries = [self.summaries[i] for i in top_indices]
        logger.info(f"Retrieved {len(relevant_summaries)} relevant summaries based on entity matching")
        logger.debug(f"Top similarities for summaries: {similarities[:5]}")
        return relevant_summaries
    
    def get_session_ids_from_summaries(self, summaries: List[Dict[str, Any]]) -> List[str]:
        """Extract session IDs from summaries."""
        session_ids = []
        for summary in summaries:
            session_id = summary.get('session_id', '')
            if session_id:
                session_ids.append(session_id)
        return session_ids
    
    def format_summaries_for_prompt(self, summaries: List[Dict[str, Any]]) -> str:
        """Format summaries for inclusion in a prompt."""
        if not summaries:
            return "No relevant summaries available"
        
        formatted_lines = []
        for i, summary in enumerate(summaries):
            session_id = summary.get('session_id', f'session_{i+1}')
            session_time = summary.get('session_time', 'unknown')
            formatted_lines.append(f"Session {session_id} ({session_time}):")
            
            # Add keys if available
            if 'keys' in summary and isinstance(summary['keys'], str):
                formatted_lines.append(f"  Keys: {summary['keys']}")
            
            # Add themes from new format
            if 'context' in summary and isinstance(summary['context'], dict):
                context = summary['context']
                
                # Extract and sort theme_x keys
                theme_keys = [k for k in context.keys() if k.startswith('theme_')]
                theme_keys.sort(key=lambda x: int(x.split('_')[1]) if x.split('_')[1].isdigit() else 0)
                
                for theme_key in theme_keys:
                    theme_num = theme_key.split('_')[1]
                    summary_key = f'summary_{theme_num}'
                    
                    theme_title = context.get(theme_key, '')
                    theme_summary = context.get(summary_key, '')
                    
                    if theme_title:
                        formatted_lines.append(f"  {theme_key}: {theme_title}")
                    if theme_summary:
                        formatted_lines.append(f"    Summary: {theme_summary}")
        
        return '\n'.join(formatted_lines)
    
    def create_summary_query_prompt(self, question: str, summaries: List[Dict[str, Any]], triples: str, chunks: str) -> str:
        """Create the final query prompt with summaries, triples and chunks."""
        formatted_summaries = self.format_summaries_for_prompt(summaries)
        
        prompt = SUMMARY_QUERY_PROMPT.replace('{question}', question)
        prompt = prompt.replace('{summaries}', formatted_summaries)
        prompt = prompt.replace('{triples}', triples)
        prompt = prompt.replace('{chunks}', chunks)
        return prompt
