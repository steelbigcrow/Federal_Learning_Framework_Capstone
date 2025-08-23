from typing import Dict, Any

from ..utils.serialization import save_checkpoint


def save_client_round(state_dict: Dict[str, Any], path: str, meta: Dict[str, Any] = None) -> None:
	"""
	保存客户端轮次的模型检查点

	Args:
		state_dict: 模型状态字典
		path: 保存路径
		meta: 元数据信息
	"""
	save_checkpoint(state_dict, path, meta)


def save_global_round(state_dict: Dict[str, Any], path: str, meta: Dict[str, Any] = None) -> None:
	"""
	保存全局轮次的模型检查点

	Args:
		state_dict: 模型状态字典
		path: 保存路径
		meta: 元数据信息
	"""
	save_checkpoint(state_dict, path, meta)
