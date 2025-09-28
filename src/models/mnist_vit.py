import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
	"""Image patch embedding module, splits image into patches and converts to embedding vectors"""

	def __init__(self, image_size: int = 28, patch_size: int = 7, emb_dim: int = 128, in_chans: int = 1):
		"""
		Initialize patch embedding

		Args:
			image_size: input image size, default 28
			patch_size: image patch size, default 7
			emb_dim: embedding dimension, default 128
			in_chans: input image channels, default 1 (grayscale)
		"""
		super().__init__()
		assert image_size % patch_size == 0  # ensure image can be divided by patch size
		self.num_patches = (image_size // patch_size) * (image_size // patch_size)  # calculate number of patches
		# use convolution layer to implement patch embedding
		self.proj = nn.Conv2d(in_chans, emb_dim, kernel_size=patch_size, stride=patch_size)

	def forward(self, x):
		"""
		Forward propagation

		Args:
			x: input image with shape (B, C, H, W)

		Returns:
			patch embedding sequence with shape (B, N, E) where N is number of patches and E is embedding dimension
		"""
		x = self.proj(x)  # (B, E, H', W') convolution operation
		x = x.flatten(2).transpose(1, 2)  # (B, N, E) flatten and transpose
		return x


class VisionTransformer(nn.Module):
	"""Vision Transformer model for image classification"""

	def __init__(self, image_size: int = 28, patch_size: int = 7, emb_dim: int = 128, depth: int = 4, nhead: int = 4, mlp_ratio: float = 2.0, num_classes: int = 10):
		"""
		Initialize Vision Transformer model

		Args:
			image_size: input image size, default 28
			patch_size: image patch size, default 7
			emb_dim: embedding dimension, default 128
			depth: number of Transformer encoder layers, default 4
			nhead: number of attention heads, default 4
			mlp_ratio: MLP hidden layer expansion ratio, default 2.0
			num_classes: number of classification classes, default 10
		"""
		super().__init__()
		# image patch embedding
		self.patch_embed = PatchEmbedding(image_size, patch_size, emb_dim)
		# learnable class token
		self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
		# positional embedding (including class token position)
		self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.num_patches, emb_dim))
		# Transformer encoder layer
		encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=nhead, dim_feedforward=int(emb_dim * mlp_ratio), batch_first=True)
		# Transformer encoder
		self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
		# layer normalization
		self.norm = nn.LayerNorm(emb_dim)
		# classification head
		self.head = nn.Linear(emb_dim, num_classes)
		# initialize weights
		self._init_weights()

	def _init_weights(self):
		"""Initialize model weights"""
		nn.init.normal_(self.cls_token, std=0.02)
		nn.init.normal_(self.pos_embed, std=0.02)

	def forward(self, x):
		"""
		Forward propagation

		Args:
			x: input image with shape (B, 1, 28, 28) where B is batch size

		Returns:
			classification logits with shape (B, 10)
		"""
		# x: (B, 1, 28, 28)
		B = x.size(0)
		x = self.patch_embed(x)  # (B, N, E) image patch embedding
		cls_tokens = self.cls_token.expand(B, -1, -1)  # expand class token to batch size
		x = torch.cat((cls_tokens, x), dim=1)  # add class token at the beginning of sequence
		x = x + self.pos_embed[:, : x.size(1)]  # add positional embedding
		x = self.encoder(x)  # Transformer encoder
		x = self.norm(x[:, 0])  # extract class token and apply layer normalization
		return self.head(x)  # classification
