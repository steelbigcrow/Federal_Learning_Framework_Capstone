from typing import Tuple
import os

import torch
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import transforms
from datasets import load_dataset


class CachedMnistDataset(Dataset):
	"""预处理并缓存的MNIST数据集，避免重复转换"""
	def __init__(self, images, labels, transform=None):
		self.images = images  # 预处理后的tensor
		self.labels = labels  # tensor
		self.transform = transform

	def __len__(self):
		return len(self.images)

	def __getitem__(self, idx):
		img = self.images[idx]
		label = self.labels[idx]
		if self.transform is not None:
			img = self.transform(img)
		return img, label

	@property
	def targets(self):
		return self.labels.tolist()


def _preprocess_mnist_split(hf_split, transform):
	"""预处理HF数据集分片，转换为tensor缓存"""
	print(f"[数据] 预处理MNIST分片，样本数: {len(hf_split)}")
	images = []
	labels = []

	for i in range(len(hf_split)):
		if i % 10000 == 0:
			print(f"[DATA] Preprocessing progress: {i}/{len(hf_split)}")

		item = hf_split[i]
		img = item['image']  # PIL Image
		label = int(item['label'])

		if transform is not None:
			img = transform(img)

		images.append(img)
		labels.append(label)
	
	# 转换为tensor
	images = torch.stack(images)
	labels = torch.tensor(labels, dtype=torch.long)

	print(f"[DATA] Preprocessing completed, images shape: {images.shape}, labels shape: {labels.shape}")
	return images, labels


def get_mnist_datasets(root: str = None, use_cache: bool = True) -> torch.utils.data.Dataset:
	"""获取MNIST训练数据集，支持缓存优化"""
	cache_dir = root or "./data_cache"
	os.makedirs(cache_dir, exist_ok=True)

	train_cache_path = os.path.join(cache_dir, "mnist_train_cached.pt")

	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.1307,), (0.3081,)),
	])
	
	# 尝试加载缓存
	if use_cache and os.path.exists(train_cache_path):
		print("[DATA] Loading MNIST training dataset from cache...")
		train_data = torch.load(train_cache_path)

		train_ds = CachedMnistDataset(train_data['images'], train_data['labels'])
		print(f"[DATA] Cache loading completed, training set: {len(train_ds)}")

	else:
		print("[DATA] First time loading, downloading and preprocessing MNIST dataset from HuggingFace...")
		ds = load_dataset('mnist')
		
		# 预处理并缓存（只处理训练集）
		train_images, train_labels = _preprocess_mnist_split(ds['train'], transform)

		train_ds = CachedMnistDataset(train_images, train_labels)
		
		# 保存缓存
		if use_cache:
			print("[DATA] Saving preprocessed data to cache...")
			torch.save({'images': train_images, 'labels': train_labels}, train_cache_path)
			print(f"[DATA] Cache saving completed: {train_cache_path}")

	return train_ds
