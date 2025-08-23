import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
	"""位置编码模块，为Transformer提供位置信息"""

	def __init__(self, d_model: int, max_len: int = 5000):
		"""
		初始化位置编码

		Args:
			d_model: 模型维度
			max_len: 最大序列长度
		"""
		super().__init__()
		# 创建位置编码矩阵
		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
		pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度使用sin
		pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度使用cos
		pe = pe.unsqueeze(0)  # 添加批次维度
		self.register_buffer('pe', pe)  # 注册为buffer，不参与梯度更新

	def forward(self, x):
		"""
		前向传播，添加位置编码

		Args:
			x: 输入序列，形状为 (B, L, D)

		Returns:
			添加位置编码后的序列
		"""
		seq_len = x.size(1)
		return x + self.pe[:, :seq_len, :]  # 截取当前序列长度对应的位置编码


class TextTransformer(nn.Module):
	"""基于Transformer的文本分类模型"""

	def __init__(self, vocab_size: int, embedding_dim: int = 128, nhead: int = 4, num_layers: int = 2, dim_feedforward: int = 256, max_len: int = 256, pad_idx: int = 1, num_classes: int = 2):
		"""
		初始化Transformer文本分类模型

		Args:
			vocab_size: 词汇表大小
			embedding_dim: 词嵌入维度
			nhead: 多头注意力头数
			num_layers: Transformer编码器层数
			dim_feedforward: 前馈网络维度
			max_len: 最大序列长度
			pad_idx: 填充标记的索引
			num_classes: 分类类别数
		"""
		super().__init__()
		# 词嵌入层
		self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
		# 位置编码
		self.pos_encoder = PositionalEncoding(embedding_dim, max_len)
		# Transformer编码器层
		encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
		# Transformer编码器
		self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
		# 分类器
		self.classifier = nn.Linear(embedding_dim, num_classes)
		self.pad_idx = pad_idx

	def forward(self, x):
		"""
		前向传播

		Args:
			x: 输入文本序列，形状为 (B, L) 其中B是批次大小，L是序列长度

		Returns:
			分类 logits
		"""
		# x: (B, L)
		mask = (x == self.pad_idx)  # 创建填充掩码
		emb = self.embedding(x)  # 词嵌入
		emb = self.pos_encoder(emb)  # 添加位置编码
		enc = self.encoder(emb, src_key_padding_mask=mask)  # Transformer编码
		# 使用 [CLS]-like 池化：取第一个位置作为序列表示
		feat = enc[:, 0, :]
		return self.classifier(feat)  # 分类
