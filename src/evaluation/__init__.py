"""
Evaluation module for federated learning framework.

This module provides comprehensive evaluation functionality for both standard
and LoRA-based federated learning models, including visualization and metrics.
"""

from .evaluator import ModelEvaluator
from .visualizer import ResultVisualizer
from .loaders import get_test_loader
from .metrics import evaluate_model


def evaluate_checkpoint(arch_config_path: str, checkpoint_path: str, 
                       lora_checkpoint_path: str = None, output_dir: str = None, 
                       device: str = None, **kwargs):
    """
    Convenient function to evaluate a checkpoint directly.
    
    Replaces the functionality of the standalone eval_final.py script.
    
    Args:
        arch_config_path: Path to architecture configuration file
        checkpoint_path: Path to main model checkpoint
        lora_checkpoint_path: Path to LoRA adapter checkpoint (optional)
        output_dir: Directory to save evaluation results (auto-generate if None)
        device: Device to run evaluation on (auto-detect if None)
        **kwargs: Additional arguments (e.g., misclassified_limit)
        
    Returns:
        Dictionary containing evaluation results
    """
    evaluator = ModelEvaluator(device=device)
    return evaluator.evaluate(
        checkpoint_path=checkpoint_path,
        arch_config_path=arch_config_path,
        lora_checkpoint_path=lora_checkpoint_path,
        output_dir=output_dir,
        **kwargs
    )


def auto_evaluate_training(arch_config_path: str, train_config_path: str,
                          use_lora: bool = False, device: str = None, outputs_root: str = None, **kwargs) -> bool:
    """
    Convenient function to automatically evaluate after training.
    
    Simplifies the evaluation call in fed_train.py.
    
    Args:
        arch_config_path: Path to architecture configuration file
        train_config_path: Path to training configuration file
        use_lora: Whether LoRA was used in training
        device: Device to run evaluation on (auto-detect if None)
        outputs_root: Root directory for outputs (default: "./outputs")
        **kwargs: Additional arguments (e.g., run_name, started_at)
        
    Returns:
        True if evaluation completed successfully, False otherwise
    """
    evaluator = ModelEvaluator(device=device, outputs_root=outputs_root or "./outputs")
    return evaluator.auto_evaluate_after_training(
        train_config_path=train_config_path,
        arch_config_path=arch_config_path,
        use_lora=use_lora,
        **kwargs
    )


__all__ = [
    'ModelEvaluator',
    'ResultVisualizer', 
    'get_test_loader',
    'evaluate_model',
    'evaluate_checkpoint',
    'auto_evaluate_training'
]