from typing import List, Dict, Any
import re
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from init.logger import logger
from init.config import ChunkConfig
from chunking.dialog_chunk_processor import DialogChunkProcessor

class ChunkProcessor:
    """Processor for document chunking operations."""

    def __init__(self, config: ChunkConfig, data_type: str = "LongmemEval"):
        """Initialize chunk processor.
        
        Args:
            config: Chunking configuration
            data_type: Type of dataset ("LongmemEval" or "LOCOMO")
        """
        self.config = config
        self.data_type = data_type
        self.dialog_processor = DialogChunkProcessor(config, data_type=data_type)
        logger.info(f"Chunk Processor initialized for {data_type} dataset")

    def chunk_by_token_size(self, text: str) -> List[Dict[str, Any]]:
        """Chunk text by token size."""
        if not text:
            return []

        # Simple word-based tokenization (for demo)
        words = text.split()
        chunks = []

        chunk_size = self.config.chunk_token_size
        overlap = self.config.chunk_overlap_token_size

        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            if chunk_words:
                chunk_text = ' '.join(chunk_words)
                chunk = {
                    'text': chunk_text,
                    'start_idx': i,
                    'end_idx': i + len(chunk_words),
                    'token_count': len(chunk_words)
                }
                chunks.append(chunk)

        logger.info(f"Created {len(chunks)} chunks from text")
        return chunks

    def chunk_by_dialog_turns(self, session_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk dialog data by conversation turns."""
        return self.dialog_processor.create_dialog_chunks(session_data)


    def chunk_document(self, doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk a single document using the configured chunking method."""
        # Check dialogue_input first, then fall back to chunk_method
        dialogue_input = getattr(self.config, 'dialogue_input', False)
        
        if dialogue_input:
            # Dialogue-based chunking
            chunks = self.chunk_by_dialog_turns(doc)
            for chunk in chunks:
                chunk['doc_id'] = doc.get('doc_id', 0)
            return chunks

        else:
            # Default: token-based chunking
            title = doc.get('title', '')
            content = doc.get('content', '')
            full_text = f"{title}\n\n{content}" if title else content
            chunks = self.chunk_by_token_size(full_text)
            for chunk in chunks:
                chunk.update({
                    'doc_id': doc.get('doc_id', 0),
                    'title': title})
            return chunks

    def process_corpus(self, corpus: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process entire corpus into chunks."""
        all_chunks = []
        global_chunk_id = 0  

        for doc in corpus:
            doc_chunks = self.chunk_document(doc)

            for chunk in doc_chunks:
                chunk['chunk_id'] = global_chunk_id 
                global_chunk_id += 1
            
            all_chunks.extend(doc_chunks)

        logger.info(f"Processed {len(corpus)} documents into {len(all_chunks)} chunks with unique chunk IDs")
        return all_chunks
