import torch.nn as nn


class MnistMLP(nn.Module):
	"""用于MNIST手写数字识别的多层感知机模型"""

	def __init__(self, input_size: int = 784, hidden_sizes=None, num_classes: int = 10):
		"""
		初始化MLP模型

		Args:
			input_size: 输入特征维度，默认为784 (28*28)
			hidden_sizes: 隐藏层大小列表，默认为[512, 256]
			num_classes: 分类类别数，默认为10 (数字0-9)
		"""
		super().__init__()
		if hidden_sizes is None:
			hidden_sizes = [512, 256]

		# 动态构建网络层
		layers = []
		in_dim = input_size
		for h in hidden_sizes:
			layers.append(nn.Linear(in_dim, h))  # 全连接层
			layers.append(nn.ReLU(inplace=True))  # ReLU激活函数
			in_dim = h
		layers.append(nn.Linear(in_dim, num_classes))  # 输出层

		# 使用Sequential组织所有层
		self.mlp = nn.Sequential(*layers)

	def forward(self, x):
		"""
		前向传播

		Args:
			x: 输入图像，形状为 (B, 1, 28, 28) 其中B是批次大小

		Returns:
			分类 logits，形状为 (B, 10)
		"""
		# x: (B, 1, 28, 28)
		b = x.size(0)
		x = x.view(b, -1)  # 展平图像为向量 (B, 784)
		return self.mlp(x)  # MLP前向传播
