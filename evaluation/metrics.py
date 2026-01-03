from typing import Dict
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from init.logger import logger

class MetricsCalculator:

    @staticmethod
    def calculate_exact_match(predicted: str, expected: str) -> float:
        if not predicted or not expected:
            return 0.0

        if expected.lower().strip() in predicted.lower().strip():
            return 1.0
        else:
            return 0.0

    @staticmethod
    def calculate_basic_metrics(predicted: str, expected: str) -> Dict[str, float]:
        exact_match = MetricsCalculator.calculate_exact_match(predicted, expected)
        
        return {
            'exact_match': exact_match,
            'has_answer': 1.0 if predicted.strip() else 0.0
        }
