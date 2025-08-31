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


def adalora_fedavg(adalora_state_dicts: List[Dict[str, torch.Tensor]], num_samples: List[int], trainable_keys: Set[str] = None) -> Dict[str, torch.Tensor]:
	"""
	AdaLoRA专用联邦平均算法，只聚合可训练的AdaLoRA权重和分类头

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

	for k in trainable_keys:
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
