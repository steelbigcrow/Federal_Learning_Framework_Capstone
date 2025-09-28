import torch.nn as nn


class MnistMLP(nn.Module):
	"""Multi-layer perceptron model for MNIST handwritten digit recognition"""

	def __init__(self, input_size: int = 784, hidden_sizes=None, num_classes: int = 10):
		"""
		Initialize MLP model

		Args:
			input_size: input feature dimension, default 784 (28*28)
			hidden_sizes: list of hidden layer sizes, default [200, 200]
			num_classes: number of classification classes, default 10 (digits 0-9)
		"""
		super().__init__()
		if hidden_sizes is None:
			hidden_sizes = [200, 200]

		# dynamically build network layers
		layers = []
		in_dim = input_size
		for h in hidden_sizes:
			layers.append(nn.Linear(in_dim, h))  # fully connected layer
			layers.append(nn.ReLU(inplace=True))  # ReLU activation function
			in_dim = h
		layers.append(nn.Linear(in_dim, num_classes))  # output layer


		# 784,200; 200,200; 200,10

		# organize all layers using Sequential
		self.mlp = nn.Sequential(*layers)

	def forward(self, x):
		"""
		Forward propagation

		Args:
			x: input image with shape (B, 1, 28, 28) where B is batch size

		Returns:
			classification logits with shape (B, 10)
		"""
		# x: (B, 1, 28, 28)
		b = x.size(0)
		x = x.view(b, -1)  # flatten image to vector (B, 784)
		return self.mlp(x)  # MLP forward propagation
