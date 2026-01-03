import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from .logger import logger

@dataclass
class ChunkConfig:
    """Configuration for document chunking."""
    chunk_token_size: int = 1200
    chunk_overlap_token_size: int = 100
    token_model: str = "gpt-3.5-turbo"
    chunk_method: str = "chunking_by_token_size"
    dialogue_input: bool = False

@dataclass
class GraphConfig:
    """Configuration for graph building."""
    graph_type: str = "dynamic_memory"
    force: bool = False
    add: bool = False
    extract_two_step: bool = True
    max_gleaning: int = 1
    enable_entity_description: bool = False
    enable_entity_type: bool = True
    enable_edge_description: bool = False
    enable_edge_name: bool = True
    enable_edge_weight: bool = True
    enable_edge_keywords: bool = False
    prior_prob: float = 0.8
    similarity_max: float = 1.0
    enable_multi_relationships: bool = True
    enable_incremental_update: bool = True
    entity_merge_threshold: float = 0.85
    relationship_merge_threshold: float = 0.9

@dataclass
class RetrieverConfig:
    """Configuration for retrieval."""
    query_type: str = "ppr"
    enable_local: bool = False
    use_entity_similarity_for_ppr: bool = False
    node_specificity: bool = True
    damping: float = 0.1
    top_k: int = 8
    top_k_triples: int = 8
    top_chunks: int = 3                   # Number of top chunks to retrieve based on triple ranking
    max_token_for_local_context: int = 4800
    enable_summary: bool = False          # Enable session summary-based retrieval
    top_summary: int = 3                  # Number of top session summaries to retrieve
    summary_weight: float = 0.3           # Weight for summary ranking in reranker (0.0-1.0)
    enable_visual: bool = False           # Enable visualization for query results
    top_visual: int = 10                  # Number of top nodes to visualize
    enable_full: bool = True              # If true, return full chunk content; if false, return only user utterances
    enable_sessiontime: bool = False      # If true, prepend session_time to chunk content in prompt
    enable_CogniRank: bool = False        # Enable CogniRank (hierarchical temporal-semantic) vs SimpleRank (weighted)
    rerank_k: float = 0.5                 # Temporal decay exponent for CogniRank (0 < k < 1)

@dataclass
class QueryConfig:
    """Configuration for query processing."""
    query_type: str = "qa"
    only_need_context: bool = False
    enable_hybrid_query: bool = True
    augmentation_ppr: bool = False
    response_type: str = "Multiple Paragraphs"
    level: int = 2
    community_information: bool = True
    retrieve_top_k: int = 20
    naive_max_token_for_text_unit: int = 12000
    local_max_token_for_text_unit: int = 4000
    max_token_for_text_unit: int = 4000
    max_token_for_global_context: int = 4000
    max_token_for_local_context: int = 4000
    entities_max_tokens: int = 2000
    relationships_max_tokens: int = 2000
    max_ir_steps: int = 2

@dataclass
class StorageConfig:
    """Configuration for storage."""
    storage_type: str = "networkx"
    persist_format: str = "pickle"
    enable_backup: bool = True
    backup_retention: int = 5

@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""
    enable_llm_eval: bool = False         # Enable LLM-based evaluation
    eval_model: str = 'gpt-4o-mini'       # Model for evaluation
    eval_temperature: float = 0.0         # Temperature for evaluation model
    eval_max_tokens: int = 10             # Max tokens for evaluation response

@dataclass
class EmbeddingConfig:
    """Configuration for embeddings."""
    api_type: str = "hf"  # hf, openai, ollama, etc.
    api_key: str = "your_apikey"
    model: str = "BAAI/bge-m3"
    cache_dir: str = "/your/cachedirectory"
    dimensions: int = 1024  # Matches your chunking model dimension
    max_token_size: int = 8102
    embed_batch_size: int = 128
    embedding_func_max_async: int = 16

@dataclass
class LLMConfig:
    """Configuration for LLM (used for graph building)."""
    api_type: str = "openai"
    api_key: str = "demo-key"  # Use demo-key for mock functionality
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-3.5-turbo"
    max_token: int = 4096
    temperature: float = 0.0
    # Concurrency control parameters
    enable_concurrent: bool = True
    max_concurrent: int = 16
    timeout: int = 600

@dataclass
class QueryLLMConfig:
    """Configuration for Query/Inference LLM (independent query stage LLM configuration)."""
    api_type: str = "openai"
    api_key: str = ""  # If empty, will use main llm config
    base_url: str = ""  # If empty, will use main llm config
    model: str = ""  # If empty, will use main llm config
    max_token: int = 0  # If 0, will use main llm config
    temperature: float = -1.0  # If negative, will use main llm config
    timeout: int = 0  # If 0, will use main llm config

@dataclass
class Config:
    """Main configuration class."""
    # Basic settings
    use_entities_vdb: bool = True
    use_relations_vdb: bool = False
    llm_model_max_token_size: int = 32768
    use_entity_link_chunk: bool = True
    enable_graph_augmentation: bool = True

    # Data settings
    index_name: str = "dynamic_memory_graph"
    vdb_type: str = "vector"
    data_type: str = "LongmemEval"  # Dataset type: "LongmemEval" or "LOCOMO"

    # Paths
    working_dir: str = "./"
    exp_name: str = "dynamic_memory"
    data_root: str = "/your/dataroot"  # Use actual data path
    dataset_name: str = "multihop-rag"

    # Embedding configuration (similar to original project)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)

    # LLM configuration (similar to original project)
    llm: LLMConfig = field(default_factory=LLMConfig)
    query_llm: QueryLLMConfig = field(default_factory=QueryLLMConfig)

    # Component configs
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    query: QueryConfig = field(default_factory=QueryConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @classmethod
    def parse(cls, config_path: Path, dataset_name: str = None) -> 'Config':
        """Parse configuration from YAML file."""
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)

        # Create config instance
        config = cls()

        # Update basic settings
        for key, value in config_dict.items():
            if hasattr(config, key):
                if isinstance(value, dict):
                    # Handle nested configs
                    sub_config = getattr(config, key)
                    for sub_key, sub_value in value.items():
                        if hasattr(sub_config, sub_key):
                            setattr(sub_config, sub_key, sub_value)
                else:
                    setattr(config, key, value)

        # Override dataset name if provided
        if dataset_name:
            config.dataset_name = dataset_name

        logger.info(f"Configuration loaded from {config_path}")
        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if hasattr(value, '__dict__'):
                result[key] = value.__dict__
            else:
                result[key] = value
        return result
