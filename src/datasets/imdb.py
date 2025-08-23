from typing import Tuple
import os
import pickle

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset

from .text_utils import build_imdb_vocab_from_texts, TextToIds, CollateText


def get_imdb_splits(root: str = None, max_seq_len: int = 256, min_freq: int = 2, use_cache: bool = True):
	"""
	获取IMDB数据集，支持本地缓存

	Args:
		root: 缓存目录路径
		max_seq_len: 最大序列长度
		min_freq: 词汇表最小频率
		use_cache: 是否使用缓存

	Returns:
		train_ds, test_ds, vocab, text_to_ids, pad_idx
	"""
	if root is None:
		root = "./data_cache"
	
	os.makedirs(root, exist_ok=True)
	
	# 定义缓存文件路径
	cache_file = os.path.join(root, f"imdb_cached_seq{max_seq_len}_freq{min_freq}.pkl")
	
	# Try to load from cache
	if use_cache and os.path.exists(cache_file):
		print(f"[IMDB] Loading data from cache: {cache_file}")
		try:
			with open(cache_file, 'rb') as f:
				cached_data = pickle.load(f)

			train_ds = cached_data['train_ds']
			test_ds = cached_data['test_ds']
			vocab = cached_data['vocab']
			text_to_ids = cached_data['text_to_ids']
			pad_idx = cached_data['pad_idx']

			print(f"[IMDB] Cache loaded successfully - Train: {len(train_ds)}, Test: {len(test_ds)}, Vocab: {len(vocab)}")
			return train_ds, test_ds, vocab, text_to_ids, pad_idx

		except Exception as e:
			print(f"[IMDB] Cache loading failed: {e}, will re-download data")
	
	# Download and process data
	print("[IMDB] Downloading and processing data...")
	ds = load_dataset('imdb')
	train_ds = ds['train']
	test_ds = ds['test']
	texts = train_ds['text']

	print("[IMDB] Building vocabulary...")
	tokenizer, vocab = build_imdb_vocab_from_texts(texts, min_freq=min_freq)
	text_to_ids = TextToIds(tokenizer, vocab, max_seq_len)
	pad_idx = vocab["<pad>"] if isinstance(vocab, dict) else vocab.token_to_index["<pad>"]
	
	# Save cache
	if use_cache:
		print(f"[IMDB] Saving cache to: {cache_file}")
		try:
			cache_data = {
				'train_ds': train_ds,
				'test_ds': test_ds,
				'vocab': vocab,
				'text_to_ids': text_to_ids,
				'pad_idx': pad_idx,
				'max_seq_len': max_seq_len,
				'min_freq': min_freq
			}
			with open(cache_file, 'wb') as f:
				pickle.dump(cache_data, f)
			print("[IMDB] Cache saved successfully")
		except Exception as e:
			print(f"[IMDB] Cache saving failed: {e}")

	print(f"[IMDB] Data preparation completed - Train: {len(train_ds)}, Test: {len(test_ds)}, Vocab: {len(vocab)}")
	return train_ds, test_ds, vocab, text_to_ids, pad_idx