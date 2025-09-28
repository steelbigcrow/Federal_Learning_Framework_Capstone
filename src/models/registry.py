from typing import Any, Dict, Optional
import torch.nn as nn

# 导入各种模型类
from .mnist_mlp import MnistMLP  # MNIST multi-layer perceptron model
from .mnist_vit import VisionTransformer  # MNIST Vision Transformer model
from .imdb_rnn import TextRNN  # IMDB text RNN model
from .imdb_lstm import TextLSTM  # IMDB text LSTM model
from .imdb_transformer import TextTransformer  # IMDB text Transformer model


def create_model(dataset: str, model: str, cfg: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> nn.Module:
	"""
	Model factory function, creates corresponding neural network model based on dataset and model type

	Args:
		dataset: dataset name ('mnist' or 'imdb')
		model: model name ('mlp', 'vit', 'rnn', 'lstm', 'transformer')
		cfg: model configuration parameter dictionary
		extra: additional parameter dictionary (for passing vocab_size, etc.)

	Returns:
		nn.Module: created neural network model instance

	Raises:
		ValueError: when dataset or model type is not supported
	"""
	extra = extra or {}

	# MNIST dataset model creation
	if dataset.lower() == 'mnist':
		if model.lower() == 'mlp':
			# create multi-layer perceptron model
			input_size = cfg.get('input_size', 784)
			hidden_sizes = cfg.get('hidden_sizes', [512, 256])
			num_classes = cfg.get('num_classes', 10)
			return MnistMLP(input_size=input_size, hidden_sizes=hidden_sizes, num_classes=num_classes)
		elif model.lower() == 'vit':
			# create Vision Transformer model
			image_size = cfg.get('image_size', 28)
			patch_size = cfg.get('patch_size', 7)
			emb_dim = cfg.get('emb_dim', 128)
			depth = cfg.get('depth', 4)
			nhead = cfg.get('nhead', 4)
			mlp_ratio = cfg.get('mlp_ratio', 2.0)
			num_classes = cfg.get('num_classes', 10)
			return VisionTransformer(image_size=image_size, patch_size=patch_size, emb_dim=emb_dim, depth=depth, nhead=nhead, mlp_ratio=mlp_ratio, num_classes=num_classes)
		else:
			raise ValueError(f"Unknown MNIST model: {model}")
# IMDB dataset model creation
	elif dataset.lower() == 'imdb':
		# get text processing related parameters
		vocab_size = extra.get('vocab_size')
		pad_idx = extra.get('pad_idx', 1)
		if vocab_size is None:
			raise ValueError('IMDB models require vocab_size in extra')

		if model.lower() == 'rnn':
			# create RNN text classification model
			emb_dim = cfg.get('embedding_dim', 128)
			hidden_size = cfg.get('hidden_size', 128)
			num_layers = cfg.get('num_layers', 1)
			bidirectional = cfg.get('bidirectional', False)
			return TextRNN(vocab_size=vocab_size, embedding_dim=emb_dim, hidden_size=hidden_size, num_layers=num_layers, bidirectional=bidirectional, pad_idx=pad_idx)
		elif model.lower() == 'lstm':
			# create LSTM text classification model
			emb_dim = cfg.get('embedding_dim', 128)
			hidden_size = cfg.get('hidden_size', 128)
			num_layers = cfg.get('num_layers', 1)
			bidirectional = cfg.get('bidirectional', False)
			return TextLSTM(vocab_size=vocab_size, embedding_dim=emb_dim, hidden_size=hidden_size, num_layers=num_layers, bidirectional=bidirectional, pad_idx=pad_idx)
		elif model.lower() == 'transformer':
			# create Transformer text classification model
			emb_dim = cfg.get('embedding_dim', 128)
			nhead = cfg.get('nhead', 4)
			num_layers = cfg.get('num_layers', 2)
			hidden_dim = cfg.get('hidden_dim', 256)
			max_len = cfg.get('max_seq_len', 256)
			return TextTransformer(vocab_size=vocab_size, embedding_dim=emb_dim, nhead=nhead, num_layers=num_layers, dim_feedforward=hidden_dim, max_len=max_len, pad_idx=pad_idx)
		else:
			raise ValueError(f"Unknown IMDB model: {model}")
	else:
		raise ValueError(f"Unknown dataset: {dataset}")
