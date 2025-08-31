"""
Dataset loading and partitioning utilities for federated learning framework.

This module provides functions for loading MNIST and IMDB datasets, partitioning them
for federated learning scenarios, and text processing utilities.
"""

from .mnist import get_mnist_datasets
from .imdb import get_imdb_splits
from .partition import partition_mnist_label_shift, partition_imdb_label_shift
from .text_utils import CollateText

__all__ = [
    'get_mnist_datasets', 'get_imdb_splits',
    'partition_mnist_label_shift', 'partition_imdb_label_shift',
    'CollateText'
]