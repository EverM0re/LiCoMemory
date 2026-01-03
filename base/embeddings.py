from typing import List, Dict, Any
import numpy as np
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from init.logger import logger

class EmbeddingManager:
    """Manager for text embeddings."""

    def __init__(self, config=None):
        """Initialize embedding manager."""
        if config:
            self.api_type = config.api_type
            self.api_key = config.api_key
            self.model_name = config.model
            self.cache_dir = config.cache_dir
            self.dimensions = config.dimensions
            self.max_token_size = config.max_token_size
            self.embed_batch_size = config.embed_batch_size
            self.embedding_func_max_async = config.embedding_func_max_async
        else:
            # Default values
            self.api_type = "openai"
            self.api_key = None
            self.model_name = "text-embedding-ada-002"
            self.cache_dir = None
            self.dimensions = 1536
            self.max_token_size = 8192
            self.embed_batch_size = 100
            self.embedding_func_max_async = 10

        # Initialize GPU/CUDA device for torch operations
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device for embedding calculations: {self.device}")

        try:
            if self.api_type == "openai":
                import openai
                if self.api_key:
                    openai.api_key = self.api_key
                self.client = openai
            elif self.api_type == "hf":
                # For HuggingFace models, we might need sentence-transformers
                try:
                    from sentence_transformers import SentenceTransformer
                    # Load model and move to GPU if available
                    self.client = SentenceTransformer(self.model_name, cache_folder=self.cache_dir, device=str(self.device))
                    logger.info(f"HuggingFace model loaded on device: {self.device}")
                except ImportError:
                    logger.warning("sentence-transformers not installed, falling back to OpenAI")
                    import openai
                    self.client = openai
            else:
                import openai
                self.client = openai

            logger.info(f"Embedding Manager initialized with model: {self.model_name} (type: {self.api_type})")
        except ImportError as e:
            logger.error(f"Failed to initialize embedding client: {e}")
            self.client = None

    async def get_embeddings(self, texts: List[str], need_tensor=False) -> List[List[float]]:
        """Get embeddings for a list of texts."""
        if not self.client:
            logger.error("Embedding client not available")
            return []

        try:
            if self.api_type == "hf":
                # HuggingFace model will use GPU automatically if device is cuda
                # convert_to_tensor=True returns torch tensors, which are already on GPU
                embeddings = self.client.encode(texts, convert_to_tensor=True, device=str(self.device))
                # Convert to list for consistency with API
                logger.info(f"self device: {self.device}")
                logger.info(f"embeddings: {embeddings.shape}")
                logger.info(f"embeddings.device: {embeddings.device}")
                if need_tensor:
                    return embeddings
                return embeddings.cpu().tolist() if isinstance(embeddings, torch.Tensor) else embeddings.tolist()
            else:
                response = await self.client.Embedding.acreate(
                    input=texts,
                    model=self.model_name
                )
                embeddings = [data.embedding for data in response.data]
                return embeddings
        except Exception as e:
            logger.error(f"Failed to get embeddings: {e}")
            return []

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors using torch on GPU."""
        try:
            # Convert to torch tensors and move to GPU
            v1 = torch.tensor(vec1, dtype=torch.float32, device=self.device)
            v2 = torch.tensor(vec2, dtype=torch.float32, device=self.device)
            
            # Calculate cosine similarity using torch operations
            # cosine_sim = (v1 · v2) / (||v1|| * ||v2||)
            dot_product = torch.dot(v1, v2)
            norm_v1 = torch.linalg.norm(v1)
            norm_v2 = torch.linalg.norm(v2)
            
            similarity = dot_product / (norm_v1 * norm_v2)
            
            # Convert back to Python float
            return similarity.item()
        except Exception as e:
            logger.error(f"Failed to calculate similarity with torch: {e}")
            return 0.0
    
    def cosine_similarity_tensor(self, vec1, vec2) -> float:
        """Calculate cosine similarity between two tensors with shape (n, d) and (m, d) using torch on GPU."""
        try:
            eps = 1e-8
            norm1 = torch.linalg.norm(vec1, dim=1, keepdim=True).clamp_min(eps)
            norm2 = torch.linalg.norm(vec2, dim=1, keepdim=True).clamp_min(eps)
            vec1_normed = vec1 / norm1          
            vec2_normed = vec2 / norm2          
            similarity = torch.matmul(vec1_normed, vec2_normed.T)  
            logger.info(f"similarity shape: {similarity.shape}")
            return similarity
        except Exception as e:
            logger.error(f"Failed to calculate similarity with torch: {e}")
            return 0.0
    

    def transfer_to_tensor(self, embeddings: List[List[float]]) -> torch.Tensor:
        """Transfer a list of embeddings to a tensor on GPU."""
        return torch.tensor(embeddings, dtype=torch.float32, device=self.device)
    
    def batch_cosine_similarity(self, query_vec: List[float], candidate_vecs: List[List[float]]) -> List[float]:
        """
        Calculate cosine similarity between one query vector and multiple candidate vectors using torch on GPU.
        This is much more efficient than calling cosine_similarity in a loop.
        
        Args:
            query_vec: Query embedding vector
            candidate_vecs: List of candidate embedding vectors
            
        Returns:
            List of similarity scores
        """
        try:
            # Convert to torch tensors and move to GPU
            query = torch.tensor(query_vec, dtype=torch.float32, device=self.device)
            candidates = torch.tensor(candidate_vecs, dtype=torch.float32, device=self.device)
            
            # Normalize vectors
            query_norm = query / torch.linalg.norm(query)
            candidates_norm = candidates / torch.linalg.norm(candidates, dim=1, keepdim=True)
            
            # Calculate cosine similarity for all candidates at once
            # This is equivalent to: [dot(query, cand) / (||query|| * ||cand||) for cand in candidates]
            similarities = torch.matmul(candidates_norm, query_norm)
            
            # Convert back to Python list
            return similarities.cpu().tolist()
        except Exception as e:
            logger.error(f"Failed to calculate batch similarity with torch: {e}")
            # Fallback to individual calculations
            return [self.cosine_similarity(query_vec, vec) for vec in candidate_vecs]
