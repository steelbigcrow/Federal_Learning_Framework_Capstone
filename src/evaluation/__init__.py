"""
Evaluation module for federated learning framework.

This module provides comprehensive evaluation functionality for both standard
and LoRA-based federated learning models, including visualization and metrics.
"""

from .evaluator import ModelEvaluator
from .visualizer import ResultVisualizer
from .loaders import get_test_loader
from .metrics import evaluate_model

__all__ = [
    'ModelEvaluator',
    'ResultVisualizer', 
    'get_test_loader',
    'evaluate_model'
]