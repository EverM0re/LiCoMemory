import json
from typing import Dict, List, Any
from pathlib import Path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from init.logger import logger


class SessionMatchingEvaluator:

    def __init__(self, results_path: str, dataset_name: str):
        self.results_path = Path(results_path)
        self.dataset_name = dataset_name

    def calculate_matching_score(self, result: Dict[str, Any]) -> float:
        origin = result.get('origin', '')
        if not origin:
            logger.warning("No origin field found in result")
            return 0.0
        origin_session_ids = self._extract_session_ids_from_origin(origin)
        
        if not origin_session_ids:
            logger.warning(f"No session IDs extracted from origin: {origin}")
            return 0.0
        
        top_session_ids = result.get('top_session_ids', [])
        if not top_session_ids:
            logger.warning("No top_session_ids found in result")
            return 0.0
        
        matched_count = 0
        for origin_session_id in origin_session_ids:
            if origin_session_id in top_session_ids:
                matched_count += 1
        
        matching_score = matched_count / len(origin_session_ids) if origin_session_ids else 0.0
        
        logger.info(f"Origin sessions: {origin_session_ids}")
        logger.info(f"Top-5 sessions: {top_session_ids}")
        logger.info(f"Matched: {matched_count}/{len(origin_session_ids)} = {matching_score:.3f}")
        
        return matching_score

    def _extract_session_ids_from_origin(self, origin) -> List[str]:
        if not origin:
            return []
        
        if isinstance(origin, list):
            return [str(sid).strip() for sid in origin if sid]
        
        if isinstance(origin, str):
            if ',' in origin or ';' in origin:
                session_ids = [sid.strip() for sid in origin.replace(';', ',').split(',')]
                return [sid for sid in session_ids if sid]
            else:
                return [origin.strip()]
        
        return []

    def evaluate_all(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not results:
            logger.warning("No results to evaluate")
            return {
                'total_questions': 0,
                'average_matching_score': 0.0,
                'matching_scores': []
            }
        
        matching_scores = []
        
        for i, result in enumerate(results):
            try:
                matching_score = self.calculate_matching_score(result)
                matching_scores.append(matching_score)
                logger.info(f"Question {i+1}: matching score = {matching_score:.3f}")
            except Exception as e:
                logger.error(f"Failed to calculate matching score for question {i+1}: {e}")
                matching_scores.append(0.0)
        
        average_matching_score = sum(matching_scores) / len(matching_scores) if matching_scores else 0.0
        
        metrics = {
            'total_questions': len(results),
            'average_matching_score': average_matching_score,
            'matching_scores': matching_scores,
            'dataset': self.dataset_name
        }
        
        logger.info(f"Session Matching Evaluation: average score = {average_matching_score:.3f}")
        return metrics

    def evaluate_from_file(self) -> Dict[str, Any]:
        try:
            with open(self.results_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            return self.evaluate_all(results)
        
        except Exception as e:
            logger.error(f"Failed to evaluate from file {self.results_path}: {e}")
            return {
                'error': str(e),
                'total_questions': 0,
                'average_matching_score': 0.0
            }

