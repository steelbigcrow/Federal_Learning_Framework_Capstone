"""
Federated Learning Framework

A comprehensive PyTorch-based federated learning framework supporting
MNIST and IMDB datasets with standard and LoRA fine-tuning capabilities.
"""

__version__ = "1.0.0"

# Core components that users might need to import directly
from .models import create_model
from .federated import Server, Client

__all__ = ['create_model', 'Server', 'Client']