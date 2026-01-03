import json
from typing import Dict, List, Any, Optional
from pathlib import Path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from init.logger import logger
from init.config import Config
from .llm_evaluator import LLMEvaluator
from .session_matching_evaluator import SessionMatchingEvaluator

class Evaluator:

    def __init__(self, results_path: str, dataset_name: str, config: Optional[Config] = None):
        self.results_path = Path(results_path)
        self.dataset_name = dataset_name
        self.config = config
        
        if config and config.evaluation.enable_llm_eval:
            self.llm_evaluator = LLMEvaluator(config, results_path, dataset_name)
            logger.info(f"Evaluator initialized with LLM evaluation for dataset: {dataset_name}")
        else:
            self.llm_evaluator = None
            logger.info(f"Evaluator initialized with exact matching for dataset: {dataset_name}")
        
        self.session_matching_evaluator = SessionMatchingEvaluator(results_path, dataset_name)

    async def evaluate(self) -> Dict[str, Any]:
        try:
            with open(self.results_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            if self.llm_evaluator:
                answer_metrics = await self.llm_evaluator.evaluate()
            else:
                answer_metrics = self._calculate_metrics(results)
            
            matching_metrics = self.session_matching_evaluator.evaluate_all(results)
            
            metrics = {
                **answer_metrics,
                'average_matching_score': matching_metrics.get('average_matching_score', 0.0),
                'matching_scores': matching_metrics.get('matching_scores', [])
            }

            logger.info(f"Evaluation completed: accuracy={metrics.get('accuracy', 0.0):.3f}, matching={metrics.get('average_matching_score', 0.0):.3f}")
            return metrics

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {"error": str(e)}

    def _calculate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not results:
            return {"error": "No results to evaluate"}

        total_questions = len(results)
        correct_answers = 0
        answered_questions = 0

        for result in results:
            raw_expected_answer = result.get('answer', '')
            expected_answer = str(raw_expected_answer).strip() if raw_expected_answer is not None else ''
            
            model_output = str(result.get('output', '')).strip()
            
            if not expected_answer:
                logger.warning(f"No expected answer found for question: {result.get('question', '')[:50]}...")
                continue
                
            if model_output:
                answered_questions += 1
                is_correct = self._check_answer_match(expected_answer, model_output)
                
                if is_correct:
                    correct_answers += 1
                    logger.debug(f"✅ Correct answer found: '{expected_answer}' in '{model_output}'")
                else:
                    logger.debug(f"❌ Incorrect answer: expected '{expected_answer}', got '{model_output}'")

        accuracy = correct_answers / total_questions if total_questions > 0 else 0
        answer_rate = answered_questions / total_questions if total_questions > 0 else 0

        metrics = {
            'total_questions': total_questions,
            'answered_questions': answered_questions,
            'correct_answers': correct_answers,
            'accuracy': accuracy,
            'answer_rate': answer_rate,
            'dataset': self.dataset_name
        }

        logger.info(f"Evaluation metrics: {correct_answers}/{total_questions} correct (accuracy: {accuracy:.3f})")
        return metrics

    def _check_answer_match(self, expected_answer: str, model_output: str) -> bool:
        if not expected_answer or not model_output:
            return False
        
        expected_lower = expected_answer.lower().strip()
        output_lower = model_output.lower().strip()
        
        # Strategy 1: Direct substring match (case-insensitive)
        if expected_lower in output_lower:
            return True
        
        # Strategy 2: Numeric answer matching
        if self._is_numeric_answer(expected_answer):
            return self._check_numeric_match(expected_answer, model_output)
        
        # Strategy 3: Normalized text matching (remove punctuation, extra spaces)
        expected_normalized = self._normalize_text(expected_answer)
        output_normalized = self._normalize_text(model_output)
        
        if expected_normalized in output_normalized:
            return True
        
        # Strategy 4: Word-level matching for short answers
        if len(expected_answer.split()) <= 3:
            expected_words = set(expected_normalized.split())
            output_words = set(output_normalized.split())
            if expected_words.issubset(output_words):
                return True
        
        return False
    
    def _is_numeric_answer(self, answer: str) -> bool:
        try:
            float(answer.strip())
            return True
        except ValueError:  
            import re
            numeric_pattern = r'^\d+(\.\d+)?\s*\w*$'
            return bool(re.match(numeric_pattern, answer.strip()))
    
    def _check_numeric_match(self, expected_answer: str, model_output: str) -> bool:
        import re
        
        expected_match = re.search(r'(\d+(?:\.\d+)?)', expected_answer)
        if not expected_match:
            return False
        expected_num = expected_match.group(1)
        output_numbers = re.findall(r'\d+(?:\.\d+)?', model_output)
        
        return expected_num in output_numbers
    
    def _normalize_text(self, text: str) -> str:
        import re
        normalized = re.sub(r'[^\w\s]', ' ', text.lower())
        normalized = re.sub(r'\s+', ' ', normalized)
        return normalized.strip()

    def save_metrics(self, metrics: Dict[str, Any], output_path: str) -> None:
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            logger.info(f"Metrics saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")
