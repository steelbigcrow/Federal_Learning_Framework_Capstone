"""
Training and evaluation utilities for federated learning framework.

This module provides core training functions, LoRA utilities, checkpoint management,
logging utilities, and visualization tools for federated learning experiments.
"""

from .lora_utils import (
    inject_lora_modules, mark_only_lora_as_trainable, load_base_model_checkpoint,
    extract_lora_state_dict, load_lora_state_dict, save_lora_checkpoint, load_lora_checkpoint
)
from .train import train_one_epoch
from .evaluate import evaluate
from .checkpoints import save_client_round, save_global_round
from .logging_utils import write_metrics_json, write_text_log
from .plotting import plot_client_metrics, plot_all_clients_metrics

__all__ = [
    'inject_lora_modules', 'mark_only_lora_as_trainable', 'load_base_model_checkpoint',
    'extract_lora_state_dict', 'load_lora_state_dict', 'save_lora_checkpoint', 'load_lora_checkpoint',
    'train_one_epoch', 'evaluate',
    'save_client_round', 'save_global_round',
    'write_metrics_json', 'write_text_log',
    'plot_client_metrics', 'plot_all_clients_metrics'
]