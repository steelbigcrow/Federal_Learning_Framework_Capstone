import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
	"""图像块嵌入模块，将图像分割成块并转换为嵌入向量"""

	def __init__(self, image_size: int = 28, patch_size: int = 7, emb_dim: int = 128, in_chans: int = 1):
		"""
		初始化块嵌入

		Args:
			image_size: 输入图像大小，默认为28
			patch_size: 图像块大小，默认为7
			emb_dim: 嵌入维度，默认为128
			in_chans: 输入图像通道数，默认为1 (灰度图)
		"""
		super().__init__()
		assert image_size % patch_size == 0  # 确保图像可以被块大小整除
		self.num_patches = (image_size // patch_size) * (image_size // patch_size)  # 计算块数量
		# 使用卷积层实现块嵌入
		self.proj = nn.Conv2d(in_chans, emb_dim, kernel_size=patch_size, stride=patch_size)

	def forward(self, x):
		"""
		前向传播

		Args:
			x: 输入图像，形状为 (B, C, H, W)

		Returns:
			块嵌入序列，形状为 (B, N, E) 其中N是块数量，E是嵌入维度
		"""
		x = self.proj(x)  # (B, E, H', W') 卷积操作
		x = x.flatten(2).transpose(1, 2)  # (B, N, E) 展平并转置
		return x


class VisionTransformer(nn.Module):
	"""用于图像分类的Vision Transformer模型"""

	def __init__(self, image_size: int = 28, patch_size: int = 7, emb_dim: int = 128, depth: int = 4, nhead: int = 4, mlp_ratio: float = 2.0, num_classes: int = 10):
		"""
		初始化Vision Transformer模型

		Args:
			image_size: 输入图像大小，默认为28
			patch_size: 图像块大小，默认为7
			emb_dim: 嵌入维度，默认为128
			depth: Transformer编码器层数，默认为4
			nhead: 多头注意力头数，默认为4
			mlp_ratio: MLP隐藏层扩展比例，默认为2.0
			num_classes: 分类类别数，默认为10
		"""
		super().__init__()
		# 图像块嵌入
		self.patch_embed = PatchEmbedding(image_size, patch_size, emb_dim)
		# 可学习的类别token
		self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
		# 位置嵌入（包括类别token位置）
		self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.num_patches, emb_dim))
		# Transformer编码器层
		encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=nhead, dim_feedforward=int(emb_dim * mlp_ratio), batch_first=True)
		# Transformer编码器
		self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
		# 层归一化
		self.norm = nn.LayerNorm(emb_dim)
		# 分类头
		self.head = nn.Linear(emb_dim, num_classes)
		# 初始化权重
		self._init_weights()

	def _init_weights(self):
		"""初始化模型权重"""
		nn.init.normal_(self.cls_token, std=0.02)
		nn.init.normal_(self.pos_embed, std=0.02)

	def forward(self, x):
		"""
		前向传播

		Args:
			x: 输入图像，形状为 (B, 1, 28, 28) 其中B是批次大小

		Returns:
			分类 logits，形状为 (B, 10)
		"""
		# x: (B, 1, 28, 28)
		B = x.size(0)
		x = self.patch_embed(x)  # (B, N, E) 图像块嵌入
		cls_tokens = self.cls_token.expand(B, -1, -1)  # 扩展类别token到批次大小
		x = torch.cat((cls_tokens, x), dim=1)  # 在序列开头添加类别token
		x = x + self.pos_embed[:, : x.size(1)]  # 添加位置嵌入
		x = self.encoder(x)  # Transformer编码器
		x = self.norm(x[:, 0])  # 提取类别token并进行层归一化
		return self.head(x)  # 分类
