from typing import Dict, List, Tuple, Any
import random

from torch.utils.data import Subset


def partition_mnist_label_shift(train_dataset, num_clients: int = 10, seed: int = 42) -> Dict[int, Subset]:
	"""为 MNIST 进行标签偏移划分：10 个 client，每个仅包含唯一一个数字标签。
	返回 {client_id: train_subset}，Client自行划分训练集和测试集
	"""
	rng = random.Random(seed)
	# 兼容 torchvision 或 HF 包装
	labels = getattr(train_dataset, 'targets', None)
	if labels is None:
		raise ValueError('train_dataset must expose targets for MNIST partitioning')
	label_to_indices: Dict[int, List[int]] = {i: [] for i in range(10)}
	for idx, y in enumerate(labels):
		label_to_indices[int(y)].append(idx)
	client_subsets: Dict[int, Subset] = {}
	for client_id in range(num_clients):
		label = client_id % 10
		indices = label_to_indices[label][:]
		rng.shuffle(indices)
		# 直接返回所有数据，不再预先分割
		client_subsets[client_id] = Subset(train_dataset, indices)
	return client_subsets


def partition_imdb_label_shift(train_data: Any, num_clients: int = 10, seed: int = 42) -> Dict[int, List[Tuple[int, str]]]:
	"""为 IMDB 进行标签偏移划分：client 0-4 仅负样本，5-9 仅正样本。
	输入 train_data 可为 HF 的 train split（dict 风格）。
	返回 {client_id: train_list}，元素为 (label, text) 且 label 为 0/1，Client自行划分训练集和测试集。
	"""
	# 将 HF 数据集转换为 (label, text) 列表
	def to_list(data: Any) -> List[Tuple[int, str]]:
		try:
			return [(int(data[i]['label']), data[i]['text']) for i in range(len(data))]
		except Exception:
			# 退化支持已有 tuple 形式
			return [(0 if (x[0] == 'neg' or x[0] == 0) else 1, x[1]) for x in list(data)]

	data_list = to_list(train_data)
	neg = [ex for ex in data_list if ex[0] == 0]
	pos = [ex for ex in data_list if ex[0] == 1]
	rng = random.Random(seed)
	rng.shuffle(neg)
	rng.shuffle(pos)

	def chunk(lst: List, n_chunks: int) -> List[List]:
		avg = len(lst) / n_chunks if n_chunks > 0 else 0
		chunks = []
		for i in range(n_chunks):
			start = int(round(i * avg))
			end = int(round((i + 1) * avg))
			chunks.append(lst[start:end])
		return chunks

	neg_shards = chunk(neg, 5)
	pos_shards = chunk(pos, 5)
	client_parts: Dict[int, List] = {}
	for cid in range(num_clients):
		if cid < 5:
			shard = neg_shards[cid] if cid < len(neg_shards) else []
		else:
			shard = pos_shards[cid - 5] if (cid - 5) < len(pos_shards) else []
		# 直接返回所有数据，不再预先分割
		client_parts[cid] = shard
	return client_parts
