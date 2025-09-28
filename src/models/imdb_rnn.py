import torch
import torch.nn as nn


class TextRNN(nn.Module):
	"""RNN-based text classification model"""

	def __init__(self, vocab_size: int, embedding_dim: int = 128, hidden_size: int = 128, num_layers: int = 1, bidirectional: bool = False, pad_idx: int = 1, num_classes: int = 2):
		"""
		Initialize RNN text classification model

		Args:
			vocab_size: vocabulary size
			embedding_dim: word embedding dimension
			hidden_size: RNN hidden layer size
			num_layers: number of RNN layers
			bidirectional: whether to use bidirectional RNN
			pad_idx: padding token index
			num_classes: number of classification classes
		"""
		super().__init__()
		# word embedding layer
		self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
		# RNN layer
		self.rnn = nn.RNN(embedding_dim, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
		# calculate output dimension (multiply by 2 if bidirectional)
		out_dim = hidden_size * (2 if bidirectional else 1)
		# classifier
		self.classifier = nn.Linear(out_dim, num_classes)

	def forward(self, x):
		"""
		Forward propagation

		Args:
			x: input text sequence with shape (B, L) where B is batch size and L is sequence length

		Returns:
			classification logits
		"""
		# x: (B, L)
		emb = self.embedding(x)  # (B, L, E) word embedding
		_, h_n = self.rnn(emb)   # h_n: (num_layers * num_directions, B, H) RNN forward propagation
		feat = h_n[-1]           # (B, H) take the last layer's hidden state as feature
		return self.classifier(feat)  # classification
