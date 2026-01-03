import json
import re
import hashlib
from typing import List, Dict, Any, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from init.logger import logger
from init.config import ChunkConfig

class DialogChunkProcessor:
    """Processor for dialog-based chunking operations."""

    def __init__(self, config: ChunkConfig, data_type: str = "LongmemEval"):
        """Initialize dialog chunk processor.
        
        Args:
            config: Chunking configuration
            data_type: Type of dataset ("LongmemEval" or "LOCOMO")
        """
        self.config = config
        self.data_type = data_type
        logger.info(f"Dialog Chunk Processor initialized for {data_type} dataset")

    def parse_dialog_turns(self, context_text: str) -> List[Dict[str, str]]:
        """
        Parse dialog context into individual turns.
        
        Args:
            context_text: Raw context string containing dialog turns
            
        Returns:
            List of dialog turns with speaker and content
        """
        dialog_turns = []
        
        if self.data_type == "LOCOMO":
            # LOCOMO format: "SpeakerName": "content"
            speaker_pattern = r'"([^"]+)"\s*:\s*"'
            positions = []
            for match in re.finditer(speaker_pattern, context_text):
                speaker_name = match.group(1)
                positions.append((speaker_name, match.end()))
            positions.sort(key=lambda x: x[1])
        else:
            # LongmemEval format: "user"/"assistant"
            user_pattern = r'"user"\s*:\s*"'
            assistant_pattern = r'"assistant"\s*:\s*"'
            positions = []
            for match in re.finditer(user_pattern, context_text):
                positions.append(('user', match.end()))
            for match in re.finditer(assistant_pattern, context_text):
                positions.append(('assistant', match.end()))
            positions.sort(key=lambda x: x[1])
        for i, (speaker, start_pos) in enumerate(positions):
            if i + 1 < len(positions):
                next_speaker = positions[i + 1][0]
                if self.data_type == "LOCOMO":
                    next_speaker_prefix_len = len(f'"{next_speaker}": "')
                else:
                    next_speaker_prefix_len = len(f'"{next_speaker}": "')
                next_speaker_pos = positions[i + 1][1] - next_speaker_prefix_len
                content_part = context_text[start_pos:next_speaker_pos]
                quote_pos = content_part.rfind('"')
                if quote_pos >= 0:
                    content = content_part[:quote_pos]
                else:
                    content = content_part.strip()
            else:
                content_part = context_text[start_pos:]
                quote_pos = content_part.find('"')
                if quote_pos >= 0:
                    content = content_part[:quote_pos]
                else:
                    content = content_part.strip()
            
            content = content.strip()
            content = re.sub(r'\s*"\s*"[^"]*":\s*', '', content) 
            content = re.sub(r'\s+', ' ', content)
            
            if content:
                dialog_turns.append({
                    'speaker': speaker,
                    'content': content
                })
        
        return dialog_turns

    def create_dialog_chunks(self, session_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create chunks from a single session's dialog data.
        
        Args:
            session_data: Dictionary containing session_time and context
            
        Returns:
            List of chunks, one per dialog turn pair (user + assistant)
        """
        session_time = str(session_data.get('session_time', 'unknown'))
        context = session_data.get('context', '')
        session_id = session_data.get('session_id', '')
        
        dialog_turns = self.parse_dialog_turns(context)
        chunks = []
        
        if not dialog_turns:
            logger.warning(f"No dialog turns found in session {session_id}")
            return []
        i = 0
        turn_pair_index = 0
        
        if self.data_type == "LOCOMO":
            # LOCOMO: Group consecutive turns from different speakers (2 turns per chunk)
            while i < len(dialog_turns):
                chunk_content_parts = []
                turn_speakers = []
                if i < len(dialog_turns):
                    turn1 = dialog_turns[i]
                    speaker1 = turn1['speaker']
                    chunk_content_parts.append(f"{speaker1}: {turn1['content']}")
                    turn_speakers.append(speaker1)
                    i += 1
                    if i < len(dialog_turns):
                        turn2 = dialog_turns[i]
                        speaker2 = turn2['speaker']
                        chunk_content_parts.append(f"{speaker2}: {turn2['content']}")
                        turn_speakers.append(speaker2)
                        i += 1
                
                if chunk_content_parts:
                    chunk_text = "\n".join(chunk_content_parts)
                    
                    chunk = {
                        'text': chunk_text,
                        'session_id': session_id,
                        'session_time': session_time
                    }
                    
                    chunks.append(chunk)
                    turn_pair_index += 1
        else:
            # LongmemEval: Group user-assistant pairs
            while i < len(dialog_turns):
                chunk_content_parts = []
                turn_speakers = []
                
                if i < len(dialog_turns) and dialog_turns[i]['speaker'] == 'user':
                    user_turn = dialog_turns[i]
                    chunk_content_parts.append(f"User: {user_turn['content']}")
                    turn_speakers.append('user')
                    i += 1
                    if i < len(dialog_turns) and dialog_turns[i]['speaker'] == 'assistant':
                        assistant_turn = dialog_turns[i]
                        chunk_content_parts.append(f"Assistant: {assistant_turn['content']}")
                        turn_speakers.append('assistant')
                        i += 1
                else:
                    if i < len(dialog_turns) and dialog_turns[i]['speaker'] == 'assistant':
                        assistant_turn = dialog_turns[i]
                        chunk_content_parts.append(f"Assistant: {assistant_turn['content']}")
                        turn_speakers.append('assistant')
                        i += 1
                
                if chunk_content_parts:
                    chunk_text = "\n".join(chunk_content_parts)
                    
                    chunk = {
                        'text': chunk_text,
                        'session_id': session_id,
                        'session_time': session_time
                    }
                    chunks.append(chunk)
                    turn_pair_index += 1
        
        return chunks

    def get_chunk_statistics(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about the chunks."""
        if not chunks:
            return {}
        
        total_chunks = len(chunks)
        total_tokens = sum(chunk.get('token_count', 0) for chunk in chunks)
        unique_sessions = len(set(chunk.get('session_id', '') for chunk in chunks))
        
        user_only_chunks = sum(1 for chunk in chunks 
                              if chunk.get('metadata', {}).get('has_user_turn', False) 
                              and not chunk.get('metadata', {}).get('has_assistant_turn', False))
        
        assistant_only_chunks = sum(1 for chunk in chunks 
                                   if chunk.get('metadata', {}).get('has_assistant_turn', False) 
                                   and not chunk.get('metadata', {}).get('has_user_turn', False))
        
        complete_pairs = sum(1 for chunk in chunks 
                            if chunk.get('metadata', {}).get('has_user_turn', False) 
                            and chunk.get('metadata', {}).get('has_assistant_turn', False))
        
        avg_tokens_per_chunk = total_tokens / total_chunks if total_chunks > 0 else 0
        
        return {
            'total_chunks': total_chunks,
            'total_tokens': total_tokens,
            'unique_sessions': unique_sessions,
            'avg_tokens_per_chunk': avg_tokens_per_chunk,
            'complete_dialog_pairs': complete_pairs,
            'user_only_chunks': user_only_chunks,
            'assistant_only_chunks': assistant_only_chunks,
            'avg_chunks_per_session': total_chunks / unique_sessions if unique_sessions > 0 else 0
        }

    def validate_chunks(self, chunks: List[Dict[str, Any]]) -> bool:
        """Validate that chunks are properly formatted."""
        required_fields = ['text', 'session_id', 'session_time', 'chunk_type']
        
        for i, chunk in enumerate(chunks):
            for field in required_fields:
                if field not in chunk:
                    logger.error(f"Chunk {i} missing required field: {field}")
                    return False
            
            if chunk.get('chunk_type') != 'dialog_turn':
                logger.error(f"Chunk {i} has invalid chunk_type: {chunk.get('chunk_type')}")
                return False
            
            if not chunk.get('text', '').strip():
                logger.error(f"Chunk {i} has empty text content")
                return False
        
        return True

