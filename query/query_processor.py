from typing import List, Dict, Any, Tuple
import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from init.logger import logger
from init.config import Config
from base.llm import LLMManager
from base.embeddings import EmbeddingManager
from prompt.query_prompt import QUERY_PROMPT, SUMMARY_QUERY_PROMPT
from utils.time_statistic import QueryTimeStatistic
from utils.cost_manager import QueryCostManager
from query.summary_retriever import SummaryRetriever
from query.visualizer import QueryResultVisualizer
from query.triple_reranker import TripleReranker

class QueryProcessor:

    @staticmethod
    def _extract_and_format_timestamp(edge_data: Dict[str, Any]) -> str:

        timestamp = edge_data.get('session_time', '') or edge_data.get('timestamp', '')
        
        if not timestamp and 'session_times' in edge_data:
            session_times = edge_data.get('session_times', [])
            valid_times = [t for t in session_times if t]
            if valid_times:
                timestamp = sorted(valid_times)[-1]
        
        if timestamp:
            try:
                date_part = timestamp.split()[0] if ' ' in timestamp else timestamp
                formatted_timestamp = date_part.replace('-', '/')
                return formatted_timestamp
            except Exception as e:
                logger.warning(f"Failed to format timestamp '{timestamp}': {e}")
                return ''
        
        return ''

    def __init__(self, config: Config, llm_manager: LLMManager, dynamic_memory=None):
        self.config = config
        self.llm = llm_manager
        self.dynamic_memory = dynamic_memory
        self.embedding_manager = EmbeddingManager(config.embedding) if hasattr(config, 'embedding') else None

        self.summary_retriever = None
        if hasattr(config.retriever, 'enable_summary') and config.retriever.enable_summary:
            self.summary_retriever = SummaryRetriever(self.embedding_manager, self.llm, self.config)

        self.visualizer = None
        if hasattr(config.retriever, 'enable_visual') and config.retriever.enable_visual:
            self.visualizer = QueryResultVisualizer(config)

        self.triple_reranker = TripleReranker(config)

        self.time_manager = QueryTimeStatistic()
        self.cost_manager = QueryCostManager(max_budget=llm_manager.cost_manager.max_budget)

    async def initialize_summary_data(self) -> bool:
        if not self.summary_retriever:
            logger.warning("Summary retriever not available")
            return False

        summaries_path = os.path.join(self.dynamic_memory.base_dir, "session_summaries.json")
        
        if not os.path.exists(summaries_path):
            logger.warning(f"Session summaries file not found at: {summaries_path}")
            return False

        self.summary_retriever.load_summaries(summaries_path)
        
        if not self.summary_retriever.summaries:
            logger.warning("No summaries loaded")
            return False
        
        await self.summary_retriever.build_summary_embeddings()
        
        if self.summary_retriever.summary_embeddings is None:
            logger.warning("Failed to build summary embeddings")
            return False
        
        logger.info(f"Successfully initialized summary data: {len(self.summary_retriever.summaries)} summaries")
        return True

    async def process_query(self, question: str, question_time: str = "") -> Dict[str, Any]:
        logger.info(f"🔍 Processing query: {question}")

        return await self._process_unified_query(question, question_time=question_time)
    
    async def _process_unified_query(self, question: str, question_time: str = "") -> Dict[str, Any]:

        self.time_manager.start_entity_extraction()
        self.time_manager.start_retrieval()
        entities = await self._extract_query_entities(question, question_time)
        self.time_manager.end_entity_extraction()

        cost_before_retrieval = self.llm.cost_manager.get_costs()
        self.time_manager.start_similar_entity_search()
        entities_with_types = await self._prepare_entities_with_types_from_extracted(entities)
        relevant_entities = await self._find_similar_entities(question, entities_with_types)
        self.time_manager.end_similar_entity_search()
        
        self.time_manager.start_triple_retrieval()
        relevant_triples = await self._get_relevant_triples(question, relevant_entities)
        self.time_manager.end_triple_retrieval()

        self.time_manager.start_summary_retrieval()
        relevant_summaries = []
        summary_rankings = {}
        
        if (self.summary_retriever and 
            hasattr(self.config.retriever, 'enable_summary') and 
            self.config.retriever.enable_summary):
            
            # Initialize summary data if needed
            if not self.summary_retriever.summaries or self.summary_retriever.summary_embeddings is None:
                if not await self.initialize_summary_data():
                    summary_rankings = {}
                    relevant_summaries = []
                else:
                    # Calculate similarity scores for all summaries and get top ones
                    summary_rankings, relevant_summaries = await self._calculate_all_summary_scores(question, entities)
            else:
                # Calculate similarity scores for all summaries and get top ones
                summary_rankings, relevant_summaries = await self._calculate_all_summary_scores(question, entities)
        else:
            summary_rankings = {}
            relevant_summaries = []
        self.time_manager.end_summary_retrieval()

        self.time_manager.start_triple_reranking()
        if relevant_triples:
            if summary_rankings:
                sorted_ranks = sorted(summary_rankings.items(), key=lambda x: x[1], reverse=True)
            
            reranked_triples = self.triple_reranker.rerank_triples(
                relevant_triples, 
                relevant_summaries, 
                summary_rankings,
                question_time=question_time
            )
            top_triples = self.triple_reranker.get_top_k_triples(reranked_triples)
        else:
            top_triples = []
        self.time_manager.end_triple_reranking()

        self.time_manager.start_chunk_retrieval()
        relevant_chunks = await self._get_chunks_for_triples(top_triples)
        self.time_manager.end_chunk_retrieval()

        formatted_prompt = await self._create_unified_prompt(
            question, top_triples, relevant_chunks, relevant_summaries, question_time=question_time
        )
        
        cost_after_retrieval = self.llm.cost_manager.get_costs()
        retrieval_prompt_tokens = cost_after_retrieval.total_prompt_tokens - cost_before_retrieval.total_prompt_tokens
        retrieval_completion_tokens = cost_after_retrieval.total_completion_tokens - cost_before_retrieval.total_completion_tokens
        self.cost_manager.update_retrieval_cost(retrieval_prompt_tokens, retrieval_completion_tokens, self.llm.model)
        self.time_manager.end_retrieval()

        self.time_manager.start_answer_generation()
        cost_before_answer = self.llm.cost_manager.get_costs()
        answer = await self._generate_answer(question, formatted_prompt)
        logger.info("🤖 LLM Answer generated")
        logger.debug(f"Answer preview: {answer[:100]}...")
        
        cost_after_answer = self.llm.cost_manager.get_costs()
        answer_prompt_tokens = cost_after_answer.total_prompt_tokens - cost_before_answer.total_prompt_tokens
        answer_completion_tokens = cost_after_answer.total_completion_tokens - cost_before_answer.total_completion_tokens
        self.cost_manager.update_answer_generation_cost(answer_prompt_tokens, answer_completion_tokens, self.llm.model)
        self.time_manager.end_answer_generation()
        
        # Get time and cost summaries
        time_summary = self.time_manager.get_query_summary()
        cost_summary = self.cost_manager.get_query_summary() if hasattr(self.cost_manager, 'get_query_summary') else {}
        
        result = {
            'question': question,
            'entities': entities,
            'relevant_entities': relevant_entities,
            'triples': top_triples,
            'chunks': relevant_chunks,
            'summaries': relevant_summaries,
            'formatted_prompt': formatted_prompt,
            'answer': answer,
            'query_summary': time_summary,
            'cost_summary': cost_summary
        }

        if self.visualizer and hasattr(self.config.retriever, 'enable_visual') and self.config.retriever.enable_visual:
            try:
                if top_triples:
                    visualization_path = self.visualizer.create_visualization(question, top_triples)
                    if visualization_path:
                        result['visualization_path'] = visualization_path
                        logger.info(f"📊 Visualization created: {visualization_path}")
            except Exception as e:
                logger.error(f"Failed to create visualization: {e}")

        logger.info(f"✅ Unified query processing completed for: {question[:70]}...")
        return result
    
    async def _prepare_entities_with_types_from_extracted(self, entities: List[str]) -> List[Dict[str, str]]:
        entities_with_types = []
        if entities:
            for entity in entities:
                entities_with_types.append({
                    'entity': entity,
                    'type': 'unknown'
                })
        
        return entities_with_types
    
    async def _create_unified_prompt(self, 
                                   question: str, 
                                   triples: List[Dict[str, Any]], 
                                   chunks: List[str], 
                                   summaries: List[Dict[str, Any]],
                                   question_time: str = "") -> str:
        if triples:
            triple_strings = [f"({triple['src']}, {triple['relation']}, {triple['tgt']})" 
                            for triple in triples]
            formatted_triples = '; '.join(triple_strings)
        else:
            formatted_triples = 'None available'
        
        if chunks:
            top_chunks = getattr(self.config.retriever, 'top_chunks', 3)
            formatted_chunks = '\n'.join([f"Chunk {i+1}: {chunk}" 
                                        for i, chunk in enumerate(chunks[:top_chunks])])
        else:
            formatted_chunks = 'None available'
        
        if summaries:
            formatted_summaries = self.summary_retriever.format_summaries_for_prompt(summaries)
            formatted_prompt = SUMMARY_QUERY_PROMPT.format(
                question_time=question_time,
                question=question,
                summaries=formatted_summaries,
                triples=formatted_triples,
                chunks=formatted_chunks
            )
        else:
            formatted_prompt = QUERY_PROMPT.format(
                question_time=question_time,
                question=question,
                triples=formatted_triples,
                chunks=formatted_chunks
            )
        
        return formatted_prompt
    
    def _log_query_summary(self):
        logger.info("=" * 80)
        logger.info("🔍 QUERY PROCESSING SUMMARY")
        logger.info("=" * 80)

        time_summary = self.time_manager.get_query_summary()

        logger.info(f"⏱️  OVERALL TIME STATISTICS:")
        logger.info(f"   🔍 Total Retrieval Time: {time_summary['retrieval_time']}s ({time_summary['retrieval_percentage']}%)")
        logger.info(f"   💬 Answer Generation Time: {time_summary['answer_generation_time']}s ({time_summary['answer_generation_percentage']}%)")
        logger.info(f"   📊 Total Query Time: {time_summary['total_query_time']}s")
        logger.info("")

        logger.info(f"🔬 DETAILED RETRIEVAL BREAKDOWN:")
        breakdown = time_summary['detailed_retrieval_breakdown']
        percentages = time_summary['retrieval_step_percentages']
        logger.info(f" Detailed Retrieval Breakdown: {breakdown}")
        steps = [
            ('1. Entity Extraction', breakdown['entity_extraction_time']),
            ('2. Similar Entity Search', breakdown['similar_entity_search_time']),
            ('3. Triple Retrieval', breakdown['triple_retrieval_time']),
            ('4. Summary Retrieval', breakdown['summary_retrieval_time']),
        ]

        logger.info(f"   {'Detailed Steps Total':30s}                : {breakdown['detailed_total']:6.2f}s")
        logger.info("")

        if hasattr(self, 'cost_manager') and self.cost_manager:
            cost_summary = self.cost_manager.get_query_summary()
            logger.info(f"💰 COST STATISTICS:")
            logger.info(f"   🔍 Retrieval Tokens: {cost_summary['retrieval_tokens']}")
            logger.info(f"   💬 Answer Generation Tokens: {cost_summary['answer_generation_tokens']}")
            logger.info(f"   📊 Total Query Tokens: {cost_summary['total_query_tokens']}")
            logger.info(f"   💵 Total Cost: ${cost_summary['total_cost_usd']:.6f}")


    async def _extract_query_entities(self, question: str, question_time: str = "") -> List[str]:
        if not question:
            return []
        try:
            entities_data = await self.llm.extract_entities(question, session_time=question_time)
            entities = [entity.get('entity', '') for entity in entities_data if entity.get('entity')]
            return entities
        except Exception as e:
            logger.error(f"Failed to extract query entities: {e}")
            return []

    async def _generate_answer(self, question: str, formatted_prompt: str) -> str:
        try:
            answer = await self.llm.generate(formatted_prompt)
            return answer.strip()
            
        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            return "Unable to generate answer at this time."

    async def _find_similar_entities(self, question: str, query_entities_with_types: List[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        if not self.dynamic_memory or not hasattr(self.dynamic_memory, 'entity_name_to_index'):
            return []
        
        if not self.embedding_manager:
            logger.warning("Embedding manager not available, falling back to string matching")
            query_entities = [e.get('entity', '') for e in (query_entities_with_types or [])]
            return await self._find_similar_entities_fallback(query_entities)
        
        all_entities = []
        if hasattr(self.dynamic_memory, 'graph_builder') and self.dynamic_memory.graph_builder.graph:
            graph = self.dynamic_memory.graph_builder.graph
            for node, data in graph.nodes(data=True):
                if 'entity_type' in data or 'entity_name' in data:
                    all_entities.append({
                        'name': node,
                        'type': data.get('entity_type', 'unknown'),
                        'description': data.get('description', '')
                    })
        else:
            for entity_name in self.dynamic_memory.entity_name_to_index.keys():
                all_entities.append({
                    'name': entity_name,
                    'type': 'unknown',
                    'description': ''
                })
        
        if not all_entities:
            return []
        
        query_entity_names = [e.get('entity', '') for e in (query_entities_with_types or [])]
        query_entity_types = [e.get('type', 'unknown') for e in (query_entities_with_types or [])]
        
        if not query_entity_names:
            query_entity_names = [question]
            query_entity_types = ['unknown']
        else:
            query_entity_names.append(question)
            query_entity_types.append('unknown')
        import time
        try:
            t1 = time.time()
            query_embeddings = await self.embedding_manager.get_embeddings(query_entity_names, need_tensor=True)
            if query_embeddings is None or len(query_embeddings) != len(query_entity_names):
                logger.error("Failed to get query embeddings, falling back to string matching")
                return await self._find_similar_entities_fallback(query_entity_names)
            entity_embeddings = []
            texts_to_embed = []
            indices_to_compute = []
            
            graph = self.dynamic_memory.graph_builder.graph if hasattr(self.dynamic_memory, 'graph_builder') else None
            
            for i, entity in enumerate(all_entities):
                entity_name = entity['name']
                
                if graph and entity_name in graph.nodes:
                    node_data = graph.nodes[entity_name]
                    if 'embedding' in node_data:
                        entity_embeddings.append(node_data['embedding'])
                    else:
                        entity_embeddings.append(None)
                        texts_to_embed.append(entity_name)
                        indices_to_compute.append(i)
                else:
                    entity_embeddings.append(None)
                    texts_to_embed.append(entity_name)
                    indices_to_compute.append(i)
            
            if texts_to_embed:
                computed_embeddings = await self.embedding_manager.get_embeddings(texts_to_embed)
                if computed_embeddings and len(computed_embeddings) == len(texts_to_embed):
                    for idx, computed_emb in zip(indices_to_compute, computed_embeddings):
                        entity_embeddings[idx] = computed_emb
            if None in entity_embeddings:
                logger.error("Some entity embeddings missing, falling back to string matching")
                return await self._find_similar_entities_fallback(query_entity_names)
            entity_embeddings = self.embedding_manager.transfer_to_tensor(entity_embeddings)
        except Exception as e:
            logger.error(f"Error getting embeddings: {e}, falling back to string matching")
            return await self._find_similar_entities_fallback(query_entity_names)
        
        t2 = time.time()
        similarities = self.embedding_manager.cosine_similarity_tensor(query_embeddings, entity_embeddings)
        entity_scores = []
        for i, entity in enumerate(all_entities):
            max_similarity = 0.0
            best_type_match = 0.0
            best_raw_similarity = 0.0
            
            for j, (query_name, query_type) in enumerate(zip(query_entity_names, query_entity_types)):
                query_name_clean = query_name.lower().strip()
                entity_name_clean = entity['name'].lower().strip()
                name_similarity = similarities[j][i]
                type_match = 1.0 if entity['type'] == query_type else 0.0
                combined_score = 0.7 * name_similarity + 0.3 * type_match
                
                if combined_score > max_similarity:
                    max_similarity = combined_score
                    best_type_match = type_match
                    best_raw_similarity = name_similarity

            final_score = max_similarity
            
            entity_scores.append({
                'name': entity['name'],
                'type': entity['type'],
                'similarity_score': best_raw_similarity,
                'type_match': best_type_match,
                'final_score': final_score
            })
        entity_scores.sort(key=lambda x: x['final_score'], reverse=True)
        top_k = self.config.retriever.top_k if hasattr(self.config, 'retriever') else 5
        
        selected_entities = entity_scores[:top_k]
        return selected_entities

    async def _find_similar_entities_fallback(self, query_entities: List[str]) -> List[Dict[str, Any]]:
        if not self.dynamic_memory or not hasattr(self.dynamic_memory, 'entity_name_to_index'):
            return []
        
        graph_entities = list(self.dynamic_memory.entity_name_to_index.keys())
        relevant_entities = []
        
        for query_entity in query_entities:
            query_entity_lower = query_entity.lower()
            for graph_entity in graph_entities:
                if query_entity_lower == graph_entity.lower():
                    if graph_entity not in [e['name'] for e in relevant_entities]:
                        relevant_entities.append({
                            'name': graph_entity,
                            'type': 'unknown',
                            'final_score': 1.0
                        })
            
            for graph_entity in graph_entities:
                if (query_entity_lower in graph_entity.lower() or 
                    graph_entity.lower() in query_entity_lower):
                    if graph_entity not in [e['name'] for e in relevant_entities]:
                        relevant_entities.append({
                            'name': graph_entity,
                            'type': 'unknown',
                            'final_score': 0.8
                        })
        
        return relevant_entities[:5]

    async def _get_relevant_triples(self, question: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self.dynamic_memory or not hasattr(self.dynamic_memory, 'graph_builder'):
            return []
        
        graph = self.dynamic_memory.graph_builder.graph
        if graph is None:
            return []
        
        entity_ranking = {}
        for i, entity in enumerate(entities):
            if isinstance(entity, dict):
                entity_name = entity.get('name', str(entity))
            else:
                entity_name = str(entity)
            entity_ranking[entity_name] = i
        
        candidate_triples = []
        for src, tgt, data in graph.edges(data=True):
            if src in entity_ranking or tgt in entity_ranking:
                triple_text = f"{src} {data.get('relation_name', 'relates_to')} {tgt}"
                
                src_rank_bonus = 1.0 / (entity_ranking.get(src, len(entities)) + 1)
                tgt_rank_bonus = 1.0 / (entity_ranking.get(tgt, len(entities)) + 1)
                entity_bonus = max(src_rank_bonus, tgt_rank_bonus)
                
                # Extract and format timestamp (latest timestamp, converted to YYYY/MM/DD format)
                timestamp = self._extract_and_format_timestamp(data)
                
                candidate_triples.append({
                    'src': src,
                    'tgt': tgt,
                    'relation': data.get('relation_name', 'relates_to'),
                    'triple_text': triple_text,
                    'chunk_id': data.get('chunk_id', ''), 
                    'chunk_ids': data.get('chunk_ids', []),
                    'session_id': data.get('session_id', ''),
                    'timestamp': timestamp,
                    'entity_bonus': entity_bonus,
                    'src_in_entities': src in entity_ranking,
                    'tgt_in_entities': tgt in entity_ranking
                })
        
        if not candidate_triples:
            logger.warning("No triples found for selected entities")
            return []
        
        logger.info(f"Found {len(candidate_triples)} candidate triples")
        
        if self.embedding_manager:
            try:
                t1 = time.time()
                question_embeddings = await self.embedding_manager.get_embeddings([question], need_tensor=True)
                logger.info(f"Time taken to get question embedding: {time.time() - t1}s")
                if question_embeddings is None:
                    raise ValueError("Failed to get question embedding")
                triple_embeddings = []
                texts_to_embed = []
                indices_to_compute = []
                
                for i, triple in enumerate(candidate_triples):
                    src = triple['src']
                    tgt = triple['tgt']
                    
                    if graph.has_edge(src, tgt):
                        edge_data = graph.edges[src, tgt]
                        if 'embedding' in edge_data:
                            triple_embeddings.append(edge_data['embedding'])
                        else:
                            triple_embeddings.append(None)
                            texts_to_embed.append(triple['triple_text'])
                            indices_to_compute.append(i)
                    else:
                        triple_embeddings.append(None)
                        texts_to_embed.append(triple['triple_text'])
                        indices_to_compute.append(i)
                
                if texts_to_embed:
                    computed_embeddings = await self.embedding_manager.get_embeddings(texts_to_embed)
                    if computed_embeddings and len(computed_embeddings) == len(texts_to_embed):
                        for idx, computed_emb in zip(indices_to_compute, computed_embeddings):
                            triple_embeddings[idx] = computed_emb
                
                triple_embeddings = self.embedding_manager.transfer_to_tensor(triple_embeddings)
                similarities = self.embedding_manager.cosine_similarity_tensor(
                    question_embeddings, triple_embeddings
                )
                
                for i, triple in enumerate(candidate_triples):
                    similarity_score = similarities[0][i]
                    
                    final_score = similarity_score + (0.2 * triple['entity_bonus'])
                    
                    triple['similarity_score'] = similarity_score
                    triple['final_score'] = final_score
            except Exception as e:
                logger.error(f"Error getting triple embeddings: {e}")
                for triple in candidate_triples:
                    triple['similarity_score'] = 0.0
                    triple['final_score'] = triple['entity_bonus']
        else:
            for triple in candidate_triples:
                triple['similarity_score'] = 0.0
                triple['final_score'] = triple['entity_bonus']

        candidate_triples.sort(key=lambda x: x['final_score'], reverse=True)
        top_k = getattr(self.config.retriever, 'top_k_triples', 10) if hasattr(self.config, 'retriever') else 10
        rerank_pool_size = max(top_k * 2, 20)
        
        selected_triples = candidate_triples[:rerank_pool_size]
        logger.info(f"Selected top {len(selected_triples)} triples for reranking (target: {top_k} final triples):")
        for i, triple in enumerate(selected_triples[:5]):
            logger.info(f"  {i+1}. {triple['triple_text']} (score: {triple['final_score']:.3f})")
        return selected_triples

    async def _get_chunks_for_triples(self, triples: List[Dict[str, Any]]) -> List[str]:
        chunks = []
        seen_chunk_ids = set()
        
        top_chunks = getattr(self.config.retriever, 'top_chunks', 3) if hasattr(self.config, 'retriever') else 3
        
        logger.debug(f"Starting chunk retrieval for {len(triples)} triples, target: {top_chunks} chunks")
        
        for triple_idx, triple in enumerate(triples):
            if len(chunks) >= top_chunks:
                logger.debug(f"Reached target of {top_chunks} chunks, stopping at triple {triple_idx}")
                break
            
            all_chunk_ids = []
            
            if 'chunk_ids' in triple and triple['chunk_ids']:
                all_chunk_ids.extend(triple['chunk_ids'])
            
            current_chunk_id = triple.get('chunk_id', '')
            if current_chunk_id and current_chunk_id not in all_chunk_ids:
                all_chunk_ids.append(current_chunk_id)
            
            if not all_chunk_ids:
                logger.debug(f"Triple {triple_idx} has no chunk_ids, skipping")
                continue
            
            session_id = triple.get('session_id', '')
            retrieved_for_this_triple = 0
            
            for chunk_id in all_chunk_ids:
                if len(chunks) >= top_chunks:
                    break
                
                chunk_id_str = str(chunk_id)
                
                if chunk_id_str in seen_chunk_ids:
                    logger.debug(f"  Chunk {chunk_id} already retrieved, skipping")
                    continue
                
                chunk_content = await self._get_chunk_content(chunk_id, session_id)
                if chunk_content:
                    chunks.append(chunk_content)
                    seen_chunk_ids.add(chunk_id_str)
                    retrieved_for_this_triple += 1
                    logger.debug(f"  ✓ Retrieved chunk {chunk_id} ({len(chunks)}/{top_chunks}) for triple: {triple.get('triple_text', '')[:50]}...")
                else:
                    logger.debug(f"  ✗ Failed to retrieve chunk {chunk_id}")
            
        return chunks

    async def _calculate_all_summary_scores(self, question: str, entities: List[str] = None) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
        try:
            all_summaries = self.summary_retriever.summaries
            if not all_summaries:
                logger.warning("No summaries available")
                return {}, []
            
            query_entities = entities or []
            if not query_entities:
                import re
                words = re.findall(r'\b\w+\b', question.lower())
                query_entities = [word for word in words if len(word) > 2]
            
            unique_texts = set()
            text_to_sessions = {}
            
            for query_entity in query_entities:
                unique_texts.add(query_entity.lower().strip())
            
            for summary in all_summaries:
                session_id = summary.get('session_id', '')
                if not session_id:
                    continue
                
                summary_keys = []
                keys_str = summary.get('keys', '')
                if keys_str and isinstance(keys_str, str):
                    summary_keys = [key.strip().lower() for key in keys_str.split(',') if key.strip()]
                
                if not summary_keys:
                    continue
                
                for summary_key in summary_keys:
                    unique_texts.add(summary_key)
                    if summary_key not in text_to_sessions:
                        text_to_sessions[summary_key] = []
                    text_to_sessions[summary_key].append(session_id)
            
            unique_texts_list = list(unique_texts)
            logger.info(f"🚀 Batch calculating embeddings for {len(unique_texts_list)} unique texts")
            all_embeddings = await self.summary_retriever.embedding_manager.get_embeddings(unique_texts_list)
            
            text_embeddings = {}
            for i, text in enumerate(unique_texts_list):
                text_embeddings[text] = all_embeddings[i]
            
            all_summary_rankings = {}
            
            for summary in all_summaries:
                session_id = summary.get('session_id', '')
                if not session_id:
                    continue
                
                summary_keys = []
                keys_str = summary.get('keys', '')
                if keys_str and isinstance(keys_str, str):
                    summary_keys = [key.strip().lower() for key in keys_str.split(',') if key.strip()]
                
                if not summary_keys:
                    all_summary_rankings[session_id] = 0.0
                    continue
                
                all_similarities = []
                
                for query_entity in query_entities:
                    query_entity_clean = query_entity.lower().strip()
                    query_embedding = text_embeddings.get(query_entity_clean)
                    
                    if query_embedding is None:
                        continue
                    
                    for summary_key in summary_keys:
                        summary_embedding = text_embeddings.get(summary_key)
                        if summary_embedding is None:
                            continue
                        
                        similarity = self.summary_retriever.embedding_manager.cosine_similarity(query_embedding, summary_embedding)
                        all_similarities.append(similarity)
                        logger.debug(f"   📊 '{query_entity_clean}' <-> '{summary_key}': {similarity:.4f}")
                
                all_similarities.sort(reverse=True)
                top_3_similarities = all_similarities[:3]
                avg_similarity = sum(top_3_similarities) / len(top_3_similarities) if top_3_similarities else 0.0
                all_summary_rankings[session_id] = avg_similarity
            
            sorted_rankings = sorted(all_summary_rankings.items(), key=lambda x: x[1], reverse=True)
            
            top_k = getattr(self.config.retriever, 'top_summary', 2) if hasattr(self.config, 'retriever') else 2
            
            top_session_ids = [session_id for session_id, _ in sorted_rankings[:top_k]]
            relevant_summaries = [summary for summary in all_summaries if summary.get('session_id') in top_session_ids]
            
            logger.info(f"✅ Selected top {len(relevant_summaries)} summaries from {len(all_summaries)} total")
            logger.info(f"📊 All summary rankings: {dict(sorted_rankings[:5])}")
            
            return all_summary_rankings, relevant_summaries
            
        except Exception as e:
            logger.warning(f"Failed to calculate all summary scores: {e}")
            relevant_summaries = await self.summary_retriever.retrieve_relevant_summaries(question)
            summary_rankings = await self._create_enhanced_summary_rankings(question, relevant_summaries, entities)
            return summary_rankings, relevant_summaries

    async def _create_enhanced_summary_rankings(self, question: str, relevant_summaries: List[Dict[str, Any]], entities: List[str] = None) -> Dict[str, float]:
        try:
            query_entities = entities or []
            if not query_entities:
                import re
                words = re.findall(r'\b\w+\b', question.lower())
                query_entities = [word for word in words if len(word) > 2]
            
            rankings = {}
            for summary in relevant_summaries:
                session_id = summary.get('session_id', '')
                if not session_id:
                    continue
                
                summary_keys = []
                keys_str = summary.get('keys', '')
                if keys_str and isinstance(keys_str, str):
                    summary_keys = [key.strip().lower() for key in keys_str.split(',') if key.strip()]
                
                if not summary_keys:
                    rankings[session_id] = 0.0
                    continue
                
                total_similarity = 0.0
                total_comparisons = 0
                
                for query_entity in query_entities:
                    query_entity_clean = query_entity.lower().strip()
                    entity_similarities = []
                    
                    for summary_key in summary_keys:
                        similarity = await self._calculate_embedding_similarity(query_entity_clean, summary_key)
                        entity_similarities.append(similarity)
                        total_similarity += similarity
                        total_comparisons += 1
                    
                avg_similarity = total_similarity / total_comparisons if total_comparisons > 0 else 0.0
                rankings[session_id] = avg_similarity
            
            return rankings
        
        except Exception as e:
            logger.warning(f"Failed to create embedding-based summary rankings: {e}")
        
        return self.triple_reranker.create_summary_rankings(relevant_summaries)
    
    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        if not name1 or not name2:
            return 0.0
        if name1 == name2:
            return 1.0
        if name1 in name2 or name2 in name1:
            return 0.8
        
        words1 = set(name1.split())
        words2 = set(name2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        jaccard_sim = intersection / union if union > 0 else 0.0
        
        if jaccard_sim > 0:
            return min(1.0, jaccard_sim + 0.2)
        
        return 0.0

    async def _calculate_embedding_similarity(self, text1: str, text2: str) -> float:
        try:
            if not text1 or not text2:
                return 0.0
            
            if not (self.summary_retriever and hasattr(self.summary_retriever, 'embedding_manager')):
                logger.warning("No embedding manager available, falling back to name similarity")
                return self._calculate_name_similarity(text1, text2)
            
            embeddings = await self.summary_retriever.embedding_manager.get_embeddings([text1, text2])
            if not embeddings or len(embeddings) != 2:
                logger.warning("Failed to get embeddings, falling back to name similarity")
                return self._calculate_name_similarity(text1, text2)
            
            embedding1, embedding2 = embeddings[0], embeddings[1]
            
            similarity = self.summary_retriever.embedding_manager.cosine_similarity(embedding1, embedding2)
            
            return float(similarity)
            
        except Exception as e:
            logger.warning(f"Failed to calculate embedding similarity between '{text1}' and '{text2}': {e}")
            return self._calculate_name_similarity(text1, text2)

    async def _get_chunk_content(self, chunk_id: str, session_id: str = '') -> str:
        chunk_content = ''
        session_time = ''
        
        if hasattr(self.dynamic_memory, 'chunk_storage') and self.dynamic_memory.chunk_storage:
            chunk_data = self.dynamic_memory.chunk_storage.get(str(chunk_id), '')
            
            if chunk_data:
                if isinstance(chunk_data, dict):
                    chunk_content = chunk_data.get('text', '')
                    session_time = chunk_data.get('session_time', '')
                else:
                    chunk_content = chunk_data
                
                if chunk_content:
                    enable_full = getattr(self.config.retriever, 'enable_full', True) if hasattr(self.config, 'retriever') else True
                    
                    if not enable_full:
                        chunk_content = self._extract_user_utterances(chunk_content)
        
        if not chunk_content and (hasattr(self.dynamic_memory, 'graph_builder') and 
            self.dynamic_memory.graph_builder.graph):
            
            graph = self.dynamic_memory.graph_builder.graph
            chunk_relations = []
            
            for src, tgt, data in graph.edges(data=True):
                if str(data.get('chunk_id', '')) == str(chunk_id):
                    relation = data.get('relation_name', '')
                    if relation:
                        chunk_relations.append(f"{src} {relation} {tgt}")
            
            if chunk_relations:
                chunk_content = f"Chunk {chunk_id}: " + "; ".join(chunk_relations[:10])
            else:
                chunk_content = f"Chunk {chunk_id}: Content related to the retrieved triples"
        
        enable_sessiontime = getattr(self.config.retriever, 'enable_sessiontime', False) if hasattr(self.config, 'retriever') else False
        
        if enable_sessiontime and session_time and chunk_content:
            chunk_content = f"{session_time} {chunk_content}"
            logger.debug(f"Prepended session_time '{session_time}' to chunk {chunk_id}")
        
        return chunk_content

    def _extract_user_utterances(self, chunk_content: str) -> str:
        if not chunk_content:
            return ""
        
        user_utterances = []
        lines = chunk_content.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('User:'):
                user_text = line[5:].strip()
                if user_text:
                    user_utterances.append(user_text)

        if not user_utterances:
            import re
            potential_user_content = []
            
            sentences = re.split(r'[.!?]+', chunk_content)
            for sentence in sentences:
                sentence = sentence.strip()
                if not any(keyword in sentence.lower() for keyword in ['assistant:', 'i can help', 'here are', 'let me', 'you should']):
                    if len(sentence) > 10:
                        potential_user_content.append(sentence)
            
            if potential_user_content:
                user_utterances.extend(potential_user_content[:2])
        
        if user_utterances:
            return ' '.join(user_utterances)
        else:
            return chunk_content[:200] + "..." if len(chunk_content) > 200 else chunk_content

    def _get_chunk_content_from_graph(self, chunk_id: str) -> str:
        if not self.dynamic_memory or not hasattr(self.dynamic_memory, 'graph_builder'):
            return ""
        
        graph = self.dynamic_memory.graph_builder.graph
        if graph is None:
            return ""
        
        chunk_content_parts = []
        for src, tgt, data in graph.edges(data=True):
            if str(data.get('chunk_id', '')) == str(chunk_id):
                relation = data.get('relation_name', '')
                if relation:
                    chunk_content_parts.append(f"{src} {relation} {tgt}")
        
        if chunk_content_parts:
            return f"Content from chunk {chunk_id}: " + "; ".join(chunk_content_parts[:3]) 
        
        return ""

    async def _get_triples_for_sessions(self, session_ids: List[str], question: str) -> List[Dict[str, Any]]:
        logger.info(f"🔍 _get_triples_for_sessions called with session_ids: {session_ids}")
        
        if not self.dynamic_memory or not hasattr(self.dynamic_memory, 'graph_builder'):
            logger.warning("No dynamic_memory or graph_builder found")
            return []
        
        graph = self.dynamic_memory.graph_builder.graph
        if graph is None:
            logger.warning("No graph found in graph_builder")
            return []
        
        logger.info(f"Graph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
        
        candidate_triples = []
        for src, tgt, data in graph.edges(data=True):
            session_id = data.get('session_id', '')
            
            belongs_to_session = False
            for target_session_id in session_ids:
                if session_id == target_session_id:
                    belongs_to_session = True
                    break
            
            if belongs_to_session:
                triple_text = f"{src} {data.get('relation_name', 'relates_to')} {tgt}"
                
                # Extract and format timestamp (latest timestamp, converted to YYYY/MM/DD format)
                timestamp = self._extract_and_format_timestamp(data)
                
                candidate_triples.append({
                    'src': src,
                    'tgt': tgt,
                    'relation': data.get('relation_name', 'relates_to'),
                    'triple_text': triple_text,
                    'chunk_id': data.get('chunk_id', ''), 
                    'chunk_ids': data.get('chunk_ids', []),
                    'session_id': session_id,
                    'timestamp': timestamp
                })
        
        logger.info(f"Found {len(candidate_triples)} candidate triples for sessions {session_ids}")
        
        if not candidate_triples:
            logger.warning(f"No triples found for sessions: {session_ids}")
            return []
        
        logger.info(f"Found {len(candidate_triples)} candidate triples for sessions")
        
        if self.embedding_manager:
            try:
                texts_to_embed = [question] + [triple['triple_text'] for triple in candidate_triples]
                embeddings = await self.embedding_manager.get_embeddings(texts_to_embed)
                
                if embeddings and len(embeddings) == len(texts_to_embed):
                    question_embedding = embeddings[0]
                    triple_embeddings = embeddings[1:]
                    
                    for i, triple in enumerate(candidate_triples):
                        similarity_score = self.embedding_manager.cosine_similarity(
                            question_embedding, triple_embeddings[i]
                        )
                        triple['similarity_score'] = similarity_score
                        triple['final_score'] = similarity_score
                        
                else:
                    logger.warning("Failed to get embeddings for session triples, using all triples")
                    for triple in candidate_triples:
                        triple['similarity_score'] = 0.0
                        triple['final_score'] = 0.0
                        
            except Exception as e:
                logger.error(f"Error getting embeddings for session triples: {e}")
                for triple in candidate_triples:
                    triple['similarity_score'] = 0.0
                    triple['final_score'] = 0.0
        else:
            for triple in candidate_triples:
                triple['similarity_score'] = 0.0
                triple['final_score'] = 0.0
        
        candidate_triples.sort(key=lambda x: x['final_score'], reverse=True)
        top_k = getattr(self.config.retriever, 'top_k', 5) if hasattr(self.config, 'retriever') else 5
        
        selected_triples = candidate_triples[:top_k]
        logger.info(f"Selected top {len(selected_triples)} triples for sessions")
        
        return selected_triples

    def _generate_triple_strings(self, relationships: List[Dict[str, Any]]) -> List[str]:
        triple_strings = []
        
        for relationship in relationships:
            src = relationship.get('src', '')
            relation = relationship.get('relation', '')
            tgt = relationship.get('tgt', '')
            
            if src and relation and tgt:
                triple_string = f"({src}, {relation}, {tgt})"
                triple_strings.append(triple_string)
        
        logger.info(f"Generated {len(triple_strings)} triple strings")
        return triple_strings
