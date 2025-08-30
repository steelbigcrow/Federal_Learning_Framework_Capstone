"""
Test data loaders for evaluation.

This module provides functions to create test data loaders for different datasets
used in the federated learning framework.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset


def get_mnist_test_loader(batch_size=512):
    """
    Create MNIST test data loader.
    
    Args:
        batch_size: Batch size for the data loader
        
    Returns:
        DataLoader for MNIST test set
    """
    ds = load_dataset("mnist", split="test")
    
    def transform(batch):
        imgs = np.stack([np.array(x, dtype="float32") / 255.0 for x in batch["image"]])[:, None, :, :]
        labels = np.array(batch["label"], dtype="int64")
        return {"x": torch.from_numpy(imgs), "y": torch.from_numpy(labels)}
    
    return DataLoader(
        ds.with_transform(transform), 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0, 
        pin_memory=True
    )


def get_imdb_test_loader(pad_idx=1, max_len=256, batch_size=128):
    """
    Create IMDB test data loader.
    
    Args:
        pad_idx: Padding index for sequences
        max_len: Maximum sequence length  
        batch_size: Batch size for the data loader
        
    Returns:
        DataLoader for IMDB test set
    """
    ds = load_dataset("imdb", split="test")
    
    def tokenize(text):
        """Simple tokenizer using hash-based vocab"""
        ids = [hash(w) % 30000 for w in text.split()][:max_len]
        if len(ids) < max_len:
            ids += [pad_idx] * (max_len - len(ids))
        return ids
    
    def transform(batch):
        X = np.stack([np.array(tokenize(t), dtype="int64") for t in batch["text"]])
        y = np.array(batch["label"], dtype="int64")
        return {"x": torch.from_numpy(X), "y": torch.from_numpy(y)}
    
    return DataLoader(
        ds.with_transform(transform), 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0, 
        pin_memory=True
    )


def get_test_loader(dataset_name, **kwargs):
    """
    Factory function to get test loader for specified dataset.
    
    Args:
        dataset_name: Name of dataset ('mnist' or 'imdb')
        **kwargs: Additional arguments passed to specific loader functions
        
    Returns:
        DataLoader for the specified dataset
        
    Raises:
        ValueError: If dataset_name is not supported
    """
    dataset_name = dataset_name.lower()
    
    if dataset_name == "mnist":
        return get_mnist_test_loader(**kwargs)
    elif dataset_name == "imdb":
        return get_imdb_test_loader(**kwargs)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")