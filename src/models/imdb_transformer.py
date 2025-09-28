import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
	"""Positional encoding module, providing position information for Transformer"""

	def __init__(self, d_model: int, max_len: int = 5000):
		"""
		Initialize positional encoding

		Args:
			d_model: model dimension
			max_len: maximum sequence length
		"""
		super().__init__()
		# create positional encoding matrix
		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
		pe[:, 0::2] = torch.sin(position * div_term)  # even dimensions use sin
		pe[:, 1::2] = torch.cos(position * div_term)  # odd dimensions use cos
		pe = pe.unsqueeze(0)  # add batch dimension
		self.register_buffer('pe', pe)  # register as buffer, no gradient update

	def forward(self, x):
		"""
		Forward propagation, add positional encoding

		Args:
			x: input sequence with shape (B, L, D)

		Returns:
			sequence with positional encoding added
		"""
		seq_len = x.size(1)
		return x + self.pe[:, :seq_len, :]  # truncate positional encoding for current sequence length


class TextTransformer(nn.Module):
	"""Transformer-based text classification model"""

	def __init__(self, vocab_size: int, embedding_dim: int = 128, nhead: int = 4, num_layers: int = 2, dim_feedforward: int = 256, max_len: int = 256, pad_idx: int = 1, num_classes: int = 2):
		"""
		Initialize Transformer text classification model

		Args:
			vocab_size: vocabulary size
			embedding_dim: word embedding dimension
			nhead: number of attention heads
			num_layers: number of Transformer encoder layers
			dim_feedforward: feedforward network dimension
			max_len: maximum sequence length
			pad_idx: padding token index
			num_classes: number of classification classes
		"""
		super().__init__()
		# word embedding layer
		self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
		# positional encoding
		self.pos_encoder = PositionalEncoding(embedding_dim, max_len)
		# Transformer encoder layer
		encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
		# Transformer encoder
		self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
		# classifier
		self.classifier = nn.Linear(embedding_dim, num_classes)
		self.pad_idx = pad_idx

	def forward(self, x):
		"""
		Forward propagation

		Args:
			x: input text sequence with shape (B, L) where B is batch size and L is sequence length

		Returns:
			classification logits
		"""
		# x: (B, L)
		mask = (x == self.pad_idx)  # create padding mask
		emb = self.embedding(x)  # word embedding
		emb = self.pos_encoder(emb)  # add positional encoding
		enc = self.encoder(emb, src_key_padding_mask=mask)  # Transformer encoding
		# use [CLS]-like pooling: take the first position as sequence representation
		feat = enc[:, 0, :]
		return self.classifier(feat)  # classification
