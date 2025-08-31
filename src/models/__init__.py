"""
Model registry and implementations for federated learning framework.

This module provides a unified interface for creating and accessing all supported models
including MNIST classifiers (MLP, Vision Transformer) and IMDB text models (RNN, LSTM, Transformer).
"""

from .registry import create_model
from .mnist_mlp import MnistMLP
from .mnist_vit import VisionTransformer
from .imdb_rnn import TextRNN
from .imdb_lstm import TextLSTM
from .imdb_transformer import TextTransformer

__all__ = [
    'create_model',
    'MnistMLP', 'VisionTransformer',
    'TextRNN', 'TextLSTM', 'TextTransformer'
]