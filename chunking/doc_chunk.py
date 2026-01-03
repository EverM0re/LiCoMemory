from typing import Dict, Any, List
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from init.logger import logger

class DocChunk:
    """Represents a document chunk with metadata."""

    def __init__(self, text: str, doc_id: int = 0, title: str = "",
                 start_idx: int = 0, end_idx: int = 0, token_count: int = 0):
        """Initialize document chunk."""
        self.text = text
        self.doc_id = doc_id
        self.title = title
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.token_count = token_count

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary."""
        return {
            'text': self.text,
            'doc_id': self.doc_id,
            'title': self.title,
            'start_idx': self.start_idx,
            'end_idx': self.end_idx,
            'token_count': self.token_count
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DocChunk':
        """Create chunk from dictionary."""
        return cls(
            text=data.get('text', ''),
            doc_id=data.get('doc_id', 0),
            title=data.get('title', ''),
            start_idx=data.get('start_idx', 0),
            end_idx=data.get('end_idx', 0),
            token_count=data.get('token_count', 0)
        )

    def __str__(self) -> str:
        """String representation."""
        return f"DocChunk(doc_id={self.doc_id}, title='{self.title[:50]}...', tokens={self.token_count})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return self.__str__()
