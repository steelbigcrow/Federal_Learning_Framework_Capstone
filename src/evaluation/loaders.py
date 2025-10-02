"""Test data loaders for evaluation.

This module provides functions to create test data loaders for different datasets
used in the federated learning framework.
"""

import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset

from ..datasets import get_imdb_splits, CollateText


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


def get_imdb_test_loader(
    pad_idx: int = 1,
    max_len: int = 256,
    batch_size: int = 128,
    cache_dir: str = "./data_cache",
    min_freq: int = 2,
    use_cache: bool = True,
):
    """
    Create IMDB test data loader.

    Args:
        pad_idx: Padding index to fall back to if cache is missing
        max_len: Maximum sequence length
        batch_size: Batch size for the data loader
        cache_dir: Directory containing cached IMDB preprocessing artefacts
        min_freq: Minimum token frequency used when building the vocabulary
        use_cache: Whether to reuse cached preprocessing data when available

    Returns:
        DataLoader for IMDB test set
    """
    cache_path = Path(cache_dir)
    cache_file = cache_path / f"imdb_cached_seq{max_len}_freq{min_freq}.pkl"

    text_to_ids = None
    cached_pad_idx = pad_idx
    test_dataset = None

    if cache_file.exists():
        try:
            with cache_file.open("rb") as f:
                cached = pickle.load(f)
            text_to_ids = cached.get("text_to_ids")
            cached_pad_idx = cached.get("pad_idx", cached_pad_idx) or cached_pad_idx
            test_dataset = cached.get("test_ds") or cached.get("test_dataset")
        except Exception as exc:
            print(f"[IMDB Eval] Failed to load cached tokenizer: {exc}; rebuilding...")

    if text_to_ids is None or test_dataset is None:
        # Fall back to rebuilding the splits so that evaluation matches training
        cache_path.mkdir(parents=True, exist_ok=True)
        _, test_dataset, _, text_to_ids, cached_pad_idx = get_imdb_splits(
            root=str(cache_path),
            max_seq_len=max_len,
            min_freq=min_freq,
            use_cache=use_cache,
        )

    # Ensure tensors are created with the same preprocessing pipeline as clients
    collate = CollateText(text_to_ids, int(cached_pad_idx), max_len)

    def collate_to_dict(batch):
        inputs, labels = collate(batch)
        return {"x": inputs, "y": labels}

    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_to_dict,
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

