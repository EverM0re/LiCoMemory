import json
import asyncio
from typing import Dict, List, Any, Optional
from pathlib import Path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from init.logger import logger
from init.config import Config
from base.llm import LLMManager

class LLMEvaluator:

    def __init__(self, config: Config, results_path: str, dataset_name: str):
        self.config = config
        self.results_path = Path(results_path)
        self.dataset_name = dataset_name
        self.enable_llm_eval = config.evaluation.enable_llm_eval
        
        if self.enable_llm_eval:
            self.eval_llm = LLMManager(
                api_key=config.llm.api_key,
                model=config.evaluation.eval_model,
                max_tokens=config.evaluation.eval_max_tokens,
                base_url=config.llm.base_url,
                enable_concurrent=False,  # Disable concurrent for evaluation stability
                max_concurrent=1,
                timeout=config.llm.timeout
            )
            self.eval_temperature = config.evaluation.eval_temperature
            self.eval_max_tokens = config.evaluation.eval_max_tokens
            logger.info(f"LLM Evaluator initialized with model: {config.evaluation.eval_model}")
        else:
            self.eval_llm = None
            logger.info("LLM evaluation disabled, using exact matching")

    def get_anscheck_prompt(self, task: str, question: str, answer: str, response: str) -> str:
        if task in ['single-session-user', 'single-session-assistant', 'multi-session']:
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == 'temporal-reasoning':
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == 'knowledge-update':
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == 'single-session-preference':
            template = "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        else:
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
            logger.warning(f"Unknown task type '{task}', using default prompt template")
        
        return prompt

    async def evaluate_with_llm(self, question: str, answer: str, response: str, question_type: str) -> bool:
        try:
            prompt = self.get_anscheck_prompt(question_type, question, answer, response)
            eval_response = await self.eval_llm.generate(
                prompt, 
                temperature=self.eval_temperature,
                max_tokens=self.eval_max_tokens
            )
            
            label = 'yes' in eval_response.lower().strip()
            logger.debug(f"LLM evaluation - Question: {question[:100]}")
            logger.debug(f"Expected: {answer}")
            logger.debug(f"Response: {response[:100]}")
            logger.debug(f"LLM says: {eval_response.strip()} -> {label}")
            
            return label
            
        except Exception as e:
            logger.error(f"LLM evaluation failed: {e}")
            return self._fallback_exact_match(answer, response)

    def _fallback_exact_match(self, expected_answer: str, model_output: str) -> bool:
        if not expected_answer or not model_output:
            return False
        
        expected_lower = expected_answer.lower().strip()
        output_lower = model_output.lower().strip()
        
        return expected_lower in output_lower

    async def evaluate(self) -> Dict[str, Any]:
        try:
            with open(self.results_path, 'r', encoding='utf-8') as f:
                results = json.load(f)

            if self.enable_llm_eval and self.eval_llm:
                metrics = await self._calculate_llm_metrics(results)
            else:
                metrics = self._calculate_exact_metrics(results)

            logger.info("Evaluation completed")
            return metrics

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {"error": str(e)}

    async def _calculate_llm_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not results:
            return {"error": "No results to evaluate"}

        total_questions = len(results)
        correct_answers = 0
        answered_questions = 0
        type_stats = {}

        logger.info(f"Starting LLM-based evaluation of {total_questions} questions...")

        for i, result in enumerate(results):
            question = result.get('question', '')
            expected_answer = str(result.get('answer', '')).strip()
            model_output = str(result.get('output', '')).strip()
            question_type = result.get('question_type', 'default')
            if question_type not in type_stats:
                type_stats[question_type] = {'total': 0, 'correct': 0}
            type_stats[question_type]['total'] += 1
            
            if not expected_answer:
                logger.warning(f"No expected answer for question {i+1}: {question[:50]}...")
                continue
                
            if model_output:
                answered_questions += 1
                is_correct = await self.evaluate_with_llm(question, expected_answer, model_output, question_type)
                
                if is_correct:
                    correct_answers += 1
                    type_stats[question_type]['correct'] += 1
                    logger.debug(f"✅ Question {i+1}: Correct")
                else:
                    logger.debug(f"❌ Question {i+1}: Incorrect")
            else:
                logger.debug(f"⚠️ Question {i+1}: No response")

            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i+1}/{total_questions} questions...")

        accuracy = correct_answers / total_questions if total_questions > 0 else 0
        answer_rate = answered_questions / total_questions if total_questions > 0 else 0

        type_accuracy = {}
        for qtype, stats in type_stats.items():
            if stats['total'] > 0:
                type_accuracy[qtype] = {
                    'accuracy': stats['correct'] / stats['total'],
                    'total': stats['total'],
                    'correct': stats['correct']
                }

        metrics = {
            'evaluation_method': 'llm_based',
            'eval_model': self.config.evaluation.eval_model,
            'total_questions': total_questions,
            'answered_questions': answered_questions,
            'correct_answers': correct_answers,
            'accuracy': accuracy,
            'answer_rate': answer_rate,
            'dataset': self.dataset_name,
            'type_accuracy': type_accuracy
        }

        logger.info(f"LLM Evaluation completed: {correct_answers}/{total_questions} correct (accuracy: {accuracy:.3f})")
        return metrics

    def _calculate_exact_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not results:
            return {"error": "No results to evaluate"}

        total_questions = len(results)
        correct_answers = 0
        answered_questions = 0

        for result in results:
            expected_answer = str(result.get('answer', '')).strip()
            model_output = str(result.get('output', '')).strip()
            
            if not expected_answer:
                continue
                
            if model_output:
                answered_questions += 1
                if self._fallback_exact_match(expected_answer, model_output):
                    correct_answers += 1

        accuracy = correct_answers / total_questions if total_questions > 0 else 0
        answer_rate = answered_questions / total_questions if total_questions > 0 else 0

        metrics = {
            'evaluation_method': 'exact_matching',
            'total_questions': total_questions,
            'answered_questions': answered_questions,
            'correct_answers': correct_answers,
            'accuracy': accuracy,
            'answer_rate': answer_rate,
            'dataset': self.dataset_name
        }

        logger.info(f"Exact matching evaluation: {correct_answers}/{total_questions} correct (accuracy: {accuracy:.3f})")
        return metrics

    def save_metrics(self, metrics: Dict[str, Any], output_path: str) -> None:
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            logger.info(f"Metrics saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")
