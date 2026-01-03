from typing import List, Dict, Any
import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from init.config import Config
from init.logger import logger
from base.llm import LLMManager
from base.embeddings import EmbeddingManager
from chunking.chunk_processor import ChunkProcessor
from .dynamic_memory import DynamicMemory

class GraphRAGCore:
    """Core GraphRAG implementation."""

    def __init__(self, config: Config, base_dir: str = "./results"):
        """Initialize GraphRAG core."""
        self.config = config
        self.base_dir = base_dir
        self.llm_manager = LLMManager(
            api_key=config.llm.api_key,
            model=config.llm.model,
            max_tokens=config.llm.max_token,
            base_url=config.llm.base_url,
            enable_concurrent=config.llm.enable_concurrent,
            max_concurrent=config.llm.max_concurrent,
            timeout=config.llm.timeout
        )
        self.embedding_manager = EmbeddingManager(config.embedding)
        data_type = getattr(config, 'data_type', 'LongmemEval')
        self.chunk_processor = ChunkProcessor(config.chunk, data_type=data_type)

        self.graph = self._create_graph(config)

        if hasattr(self.graph, 'set_extractors'):
            self.graph.set_extractors(self.llm_manager, self.embedding_manager)

        logger.info("GraphRAG Core initialized")

    def _create_graph(self, config: Config):
        graph = DynamicMemory(config, self.base_dir)
        graph.chunk_processor = self.chunk_processor
        return graph

    async def insert(self, corpus: List[Dict[str, Any]]) -> None:
        self.graph.time_manager.start_total_graph_building()
        
        add = getattr(self.config.graph, 'add', False)
        force = getattr(self.config.graph, 'force', False)
        
        if add and corpus:
            logger.info("🔄 ADD MODE: Processing sessions sequentially")
            
            graph_path = os.path.join(self.base_dir, f"{self.config.index_name}.pkl")
            if os.path.exists(graph_path):
                logger.info(f"Loading existing graph from {graph_path}")
                self.graph.load_graph(graph_path)
                stats = self.graph.get_graph_stats()
                logger.info(f"Loaded graph with {stats['num_nodes']} nodes and {stats['num_edges']} edges")
            else:
                logger.info("No existing graph found, will create new graph")
                await self.graph.build_graph([])
            
            sessions_map = {}
            for doc in corpus:
                session_id = doc.get('session_id', 'unknown')
                if session_id not in sessions_map:
                    sessions_map[session_id] = []
                sessions_map[session_id].append(doc)
            
            logger.info(f"Found {len(sessions_map)} unique sessions to process")
            
            for idx, (session_id, session_docs) in enumerate(sessions_map.items(), 1):
                logger.info(f"\n{'='*80}")
                logger.info(f"Processing session {idx}/{len(sessions_map)}: {session_id}")
                logger.info(f"{'='*80}")
                
                await self.graph.add_single_session(session_docs)
                
                self.graph.save_graph(graph_path)
                stats = self.graph.get_graph_stats()
                logger.info(f"Graph saved with {stats['num_nodes']} nodes and {stats['num_edges']} edges")
            
            logger.info("\n" + "="*80)
            logger.info("✅ All sessions processed successfully")
            logger.info("="*80)
            
        else:
            if self.config.retriever.enable_summary:
                logger.info("Generating session summaries...")
                summaries = await self.graph.generate_session_summaries(corpus)
                logger.info(f"Generated {len(summaries)} session summaries")
            else:
                logger.info("Summary generation disabled in config")
                summaries = []
            
            chunks = self.chunk_processor.process_corpus(corpus)
            logger.info(f"Processed {len(chunks)} chunks from {len(corpus)} documents")

            await self.graph.build_graph(chunks)
            
            graph_path = os.path.join(self.base_dir, f"{self.config.index_name}.pkl")
            self.graph.save_graph(graph_path)

            stats = self.graph.get_graph_stats()
            logger.info(f"Graph built with {stats['num_nodes']} nodes and {stats['num_edges']} edges")
            logger.info(f"Graph saved to {graph_path}")
        
        self.graph.time_manager.end_total_graph_building()
        logger.info("Document insertion completed")

