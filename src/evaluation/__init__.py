"""
Evaluation module for federated learning framework.

This module provides comprehensive evaluation functionality for both standard
and LoRA-based federated learning models, including visualization and metrics.
"""

from .visualizer import ResultVisualizer
from .loaders import get_test_loader
from .metrics import evaluate_model
from .round_evaluator import RoundEvaluator


__all__ = [
    'ResultVisualizer', 
    'get_test_loader',
    'evaluate_model',
    'RoundEvaluator'
]