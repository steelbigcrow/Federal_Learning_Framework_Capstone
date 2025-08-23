import torch
import torch.nn as nn


class TextLSTM(nn.Module):
	"""基于LSTM的文本分类模型"""

	def __init__(self, vocab_size: int, embedding_dim: int = 128, hidden_size: int = 128, num_layers: int = 1, bidirectional: bool = False, pad_idx: int = 1, num_classes: int = 2):
		"""
		初始化LSTM文本分类模型

		Args:
			vocab_size: 词汇表大小
			embedding_dim: 词嵌入维度
			hidden_size: LSTM隐藏层大小
			num_layers: LSTM层数
			bidirectional: 是否双向LSTM
			pad_idx: 填充标记的索引
			num_classes: 分类类别数
		"""
		super().__init__()
		# 词嵌入层
		self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
		# LSTM层
		self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
		# 计算输出维度（双向则乘2）
		out_dim = hidden_size * (2 if bidirectional else 1)
		# 分类器
		self.classifier = nn.Linear(out_dim, num_classes)

	def forward(self, x):
		"""
		前向传播

		Args:
			x: 输入文本序列，形状为 (B, L) 其中B是批次大小，L是序列长度

		Returns:
			分类 logits
		"""
		# x: (B, L)
		emb = self.embedding(x)  # 词嵌入
		_, (h_n, _) = self.lstm(emb)  # LSTM前向传播，获取最后一个隐藏状态
		feat = h_n[-1]  # 取最后一层的隐藏状态作为特征
		return self.classifier(feat)  # 分类
