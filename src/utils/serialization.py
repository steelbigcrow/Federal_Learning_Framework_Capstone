"""
序列化工具模块
"""
import json
import os
from typing import Any, Dict

import torch


def save_checkpoint(state_dict: Dict[str, Any], path: str, meta: Dict[str, Any] = None) -> None:
	"""
	保存模型检查点到文件

	Args:
		state_dict: 模型状态字典
		path: 保存路径
		meta: 元数据字典，如果提供则保存为JSON文件
	"""
	os.makedirs(os.path.dirname(path), exist_ok=True)
	torch.save(state_dict, path)
	if meta is not None:
		meta_path = path + ".json"
		with open(meta_path, 'w', encoding='utf-8') as f:
			json.dump(meta, f, ensure_ascii=False, indent=2)


def load_checkpoint(path: str, map_location: str = 'cpu') -> Dict[str, Any]:
	"""
	从文件加载模型检查点

	Args:
		path: 检查点文件路径
		map_location: 设备映射位置，默认为'cpu'

	Returns:
		加载的状态字典
	"""
	return torch.load(path, map_location=map_location)


def try_load_meta(path: str) -> Dict[str, Any]:
	"""
	尝试加载检查点的元数据文件

	Args:
		path: 检查点文件路径（不含.json扩展名）

	Returns:
		元数据字典，如果文件不存在则返回空字典
	"""
	meta_path = path + ".json"
	if os.path.exists(meta_path):
		with open(meta_path, 'r', encoding='utf-8') as f:
			return json.load(f)
	return {}
