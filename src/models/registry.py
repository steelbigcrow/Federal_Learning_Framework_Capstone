from typing import Any, Dict, Optional
import torch.nn as nn

# 导入各种模型类
from .mnist_mlp import MnistMLP  # MNIST多层感知机模型
from .mnist_vit import VisionTransformer  # MNIST Vision Transformer模型
from .imdb_rnn import TextRNN  # IMDB文本RNN模型
from .imdb_lstm import TextLSTM  # IMDB文本LSTM模型
from .imdb_transformer import TextTransformer  # IMDB文本Transformer模型


def create_model(dataset: str, model: str, cfg: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> nn.Module:
	"""
	模型工厂函数，根据数据集和模型类型创建相应的神经网络模型

	Args:
		dataset: 数据集名称 ('mnist' 或 'imdb')
		model: 模型名称 ('mlp', 'vit', 'rnn', 'lstm', 'transformer')
		cfg: 模型配置参数字典
		extra: 额外参数字典（用于传递vocab_size等）

	Returns:
		nn.Module: 创建的神经网络模型实例

	Raises:
		ValueError: 当数据集或模型类型不支持时抛出
	"""
	extra = extra or {}

	# MNIST数据集模型创建
	if dataset.lower() == 'mnist':
		if model.lower() == 'mlp':
			# 创建多层感知机模型
			input_size = cfg.get('input_size', 784)
			hidden_sizes = cfg.get('hidden_sizes', [512, 256])
			num_classes = cfg.get('num_classes', 10)
			return MnistMLP(input_size=input_size, hidden_sizes=hidden_sizes, num_classes=num_classes)
		elif model.lower() == 'vit':
			# 创建Vision Transformer模型
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
	# IMDB数据集模型创建
	elif dataset.lower() == 'imdb':
		# 获取文本处理相关参数
		vocab_size = extra.get('vocab_size')
		pad_idx = extra.get('pad_idx', 1)
		if vocab_size is None:
			raise ValueError('IMDB models require vocab_size in extra')

		if model.lower() == 'rnn':
			# 创建RNN文本分类模型
			emb_dim = cfg.get('embedding_dim', 128)
			hidden_size = cfg.get('hidden_size', 128)
			num_layers = cfg.get('num_layers', 1)
			bidirectional = cfg.get('bidirectional', False)
			return TextRNN(vocab_size=vocab_size, embedding_dim=emb_dim, hidden_size=hidden_size, num_layers=num_layers, bidirectional=bidirectional, pad_idx=pad_idx)
		elif model.lower() == 'lstm':
			# 创建LSTM文本分类模型
			emb_dim = cfg.get('embedding_dim', 128)
			hidden_size = cfg.get('hidden_size', 128)
			num_layers = cfg.get('num_layers', 1)
			bidirectional = cfg.get('bidirectional', False)
			return TextLSTM(vocab_size=vocab_size, embedding_dim=emb_dim, hidden_size=hidden_size, num_layers=num_layers, bidirectional=bidirectional, pad_idx=pad_idx)
		elif model.lower() == 'transformer':
			# 创建Transformer文本分类模型
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
