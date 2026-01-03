import re
import json
import asyncio
from typing import List, Dict, Any, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from init.logger import logger

class Utils:
    """Utility class with common functions."""

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text."""
        if not text:
            return ""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        return text

    @staticmethod
    def parse_json_from_response(response: str) -> Dict[str, Any]:
        """Parse JSON from LLM response."""
        try:
            # Try to extract JSON from response (support both objects and arrays)
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                content = json_match.group(1).strip()
                parsed = json.loads(content)
                return parsed

            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                content = json_match.group(0).strip()
                parsed = json.loads(content)
                return parsed

            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                content = json_match.group(0).strip()
                parsed = json.loads(content)
                return parsed

            logger.warning("No JSON found in response")
            return {}

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from response: {e}")
            logger.debug(f"Response content: {response[:500]}...")
            return {}

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """Split text into chunks with overlap."""
        if not text:
            return []

        words = text.split()
        chunks = []

        for i in range(0, len(words), chunk_size - overlap):
            chunk = words[i:i + chunk_size]
            if chunk:
                chunks.append(' '.join(chunk))

        return chunks

    @staticmethod
    def calculate_similarity(text1: str, text2: str) -> float:
        """Calculate simple Jaccard similarity between two texts."""
        if not text1 or not text2:
            return 0.0

        set1 = set(text1.lower().split())
        set2 = set(text2.lower().split())

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union > 0 else 0.0

    @staticmethod
    async def batch_process(items: List[Any], batch_size: int = 10,
                           processor_func=None) -> List[Any]:
        """Process items in batches asynchronously."""
        if not processor_func:
            return items

        results = []
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batch_results = await asyncio.gather(*[processor_func(item) for item in batch])
            results.extend(batch_results)

        return results
