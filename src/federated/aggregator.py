"""
联邦学习聚合算法模块
"""
from typing import Dict, List, Set
import torch


def fedavg(state_dicts: List[Dict[str, torch.Tensor]], num_samples: List[int]) -> Dict[str, torch.Tensor]:
	"""
	标准联邦平均算法，用于基模训练（聚合所有权重）

	Args:
		state_dicts: 客户端模型状态字典列表
		num_samples: 各客户端样本数量列表

	Returns:
		聚合后的全局模型状态字典
	"""
	assert len(state_dicts) == len(num_samples) and len(state_dicts) > 0
	total = float(sum(num_samples))
	new_state: Dict[str, torch.Tensor] = {}
	for k in state_dicts[0].keys():
		acc = None
		for sd, n in zip(state_dicts, num_samples):
			w = n / total
			if acc is None:
				acc = sd[k].detach() * w
			else:
				acc += sd[k].detach() * w
		new_state[k] = acc
	return new_state


def lora_fedavg(lora_state_dicts: List[Dict[str, torch.Tensor]], num_samples: List[int], trainable_keys: Set[str] = None) -> Dict[str, torch.Tensor]:
	"""
	LoRA专用联邦平均算法，只聚合可训练的LoRA权重和分类头

	Args:
		lora_state_dicts: LoRA模型状态字典列表
		num_samples: 各客户端样本数量列表
		trainable_keys: 可训练参数的键名集合，如果为None则聚合所有权重

	Returns:
		聚合后的LoRA模型状态字典
	"""
	assert len(lora_state_dicts) == len(num_samples) and len(lora_state_dicts) > 0
	total = float(sum(num_samples))
	new_state: Dict[str, torch.Tensor] = {}

	# 如果没有指定trainable_keys，则聚合所有权重（向后兼容）
	if trainable_keys is None:
		keys_to_aggregate = set(lora_state_dicts[0].keys())
	else:
		keys_to_aggregate = trainable_keys

	# 只聚合可训练的权重
	for k in lora_state_dicts[0].keys():
		if k in keys_to_aggregate:
			acc = None
			for sd, n in zip(lora_state_dicts, num_samples):
				w = n / total
				if acc is None:
					acc = sd[k].detach() * w
				else:
					acc += sd[k].detach() * w
			new_state[k] = acc
		else:
			# 对于非可训练权重，直接使用第一个客户端的权重（应该都相同）
			new_state[k] = lora_state_dicts[0][k].detach().clone()

	return new_state


def get_trainable_keys(model: torch.nn.Module) -> Set[str]:
	"""
	获取模型中所有可训练参数的键名

	Args:
		model: PyTorch模型

	Returns:
		可训练参数的键名集合
	"""
	trainable_keys = set()
	for name, param in model.named_parameters():
		if param.requires_grad:
			trainable_keys.add(name)
	return trainable_keys


def adalora_fedavg_with_padding(adalora_state_dicts: List[Dict[str, torch.Tensor]], 
                               num_samples: List[int], 
                               trainable_keys: Set[str] = None) -> Dict[str, torch.Tensor]:
	"""
	AdaLoRA专用联邦平均算法，支持不同秩的零填充聚合
	
	对于基模的某一层，不同客户端返回不同的A、B矩阵：
	- A.size = (m, r_i), B.size = (r_i, n)
	- 找到最大的r_i，对较小的矩阵进行零填充
	- A矩阵填充为(m, r_max)，B矩阵填充为(r_max, n)
	- 然后进行标准的加权平均聚合

	Args:
		adalora_state_dicts: AdaLoRA模型状态字典列表
		num_samples: 各客户端样本数量列表
		trainable_keys: 可训练参数的键名集合，如果为None则聚合所有权重

	Returns:
		聚合后的AdaLoRA模型状态字典
	"""
	assert len(adalora_state_dicts) == len(num_samples) and len(adalora_state_dicts) > 0
	
	total = float(sum(num_samples))
	new_state: Dict[str, torch.Tensor] = {}

	# 如果没有指定trainable_keys，则聚合所有权重（向后兼容）
	if trainable_keys is None:
		trainable_keys = set(adalora_state_dicts[0].keys())

	# 分组处理A和B矩阵
	lora_pairs = {}  # 存储每一层的A、B矩阵对
	other_keys = []  # 存储非LoRA参数

	# 识别LoRA参数对
	for k in trainable_keys:
		if k.endswith('.lora_A'):
			layer_prefix = k[:-7]  # 移除'.lora_A'
			if layer_prefix not in lora_pairs:
				lora_pairs[layer_prefix] = {}
			lora_pairs[layer_prefix]['A'] = k
		elif k.endswith('.lora_B'):
			layer_prefix = k[:-7]  # 移除'.lora_B'
			if layer_prefix not in lora_pairs:
				lora_pairs[layer_prefix] = {}
			lora_pairs[layer_prefix]['B'] = k
		else:
			other_keys.append(k)

	# 处理LoRA矩阵对（A和B）
	for layer_prefix, keys in lora_pairs.items():
		if 'A' in keys and 'B' in keys:
			a_key = keys['A']
			b_key = keys['B']
			
			# 检查所有客户端是否都有这个层的A、B矩阵
			if all(a_key in sd and b_key in sd for sd in adalora_state_dicts):
				# 找到最大的秩
				max_rank = 0
				for sd in adalora_state_dicts:
					a_matrix = sd[a_key]
					b_matrix = sd[b_key]
					# A矩阵的列数或B矩阵的行数就是当前的秩
					current_rank = min(a_matrix.shape[1], b_matrix.shape[0])
					max_rank = max(max_rank, current_rank)
				
				print(f"[AdaLoRA Aggregation] Layer {layer_prefix}: max_rank = {max_rank}")
				
				# 对A和B矩阵进行零填充和聚合
				a_acc = None
				b_acc = None
				
				for sd, n in zip(adalora_state_dicts, num_samples):
					w = n / total
					a_matrix = sd[a_key]
					b_matrix = sd[b_key]
					
					# 零填充A矩阵: (m, r_i) -> (m, r_max)
					current_rank_a = a_matrix.shape[1]
					if current_rank_a < max_rank:
						padding = max_rank - current_rank_a
						a_padded = torch.nn.functional.pad(a_matrix, (0, padding), 'constant', 0)
					else:
						a_padded = a_matrix
					
					# 零填充B矩阵: (r_i, n) -> (r_max, n)
					current_rank_b = b_matrix.shape[0]
					if current_rank_b < max_rank:
						padding = max_rank - current_rank_b
						b_padded = torch.nn.functional.pad(b_matrix, (0, 0, 0, padding), 'constant', 0)
					else:
						b_padded = b_matrix
					
					# 累加权重
					if a_acc is None:
						a_acc = a_padded.detach() * w
						b_acc = b_padded.detach() * w
					else:
						a_acc += a_padded.detach() * w
						b_acc += b_padded.detach() * w
				
				new_state[a_key] = a_acc
				new_state[b_key] = b_acc

	# 处理其他参数（非LoRA参数）
	for k in other_keys:
		if k in adalora_state_dicts[0]:
			acc = None
			for sd, n in zip(adalora_state_dicts, num_samples):
				w = n / total
				if acc is None:
					acc = sd[k].detach() * w
				else:
					acc += sd[k].detach() * w
			new_state[k] = acc

	return new_state


def adalora_fedavg(adalora_state_dicts: List[Dict[str, torch.Tensor]], 
                  num_samples: List[int], 
                  trainable_keys: Set[str] = None) -> Dict[str, torch.Tensor]:
	"""
	AdaLoRA专用联邦平均算法，使用零填充策略支持不同秩的聚合

	Args:
		adalora_state_dicts: AdaLoRA模型状态字典列表
		num_samples: 各客户端样本数量列表
		trainable_keys: 可训练参数的键名集合，如果为None则聚合所有权重

	Returns:
		聚合后的AdaLoRA模型状态字典
	"""
	return adalora_fedavg_with_padding(adalora_state_dicts, num_samples, trainable_keys)
