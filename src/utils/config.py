"""
配置文件加载和管理模块
"""
import argparse
import yaml
from typing import Any, Dict, List


def load_yaml(path: str) -> Dict[str, Any]:
	"""
	加载YAML配置文件

	Args:
		path: 配置文件路径

	Returns:
		解析后的配置字典，如果文件不存在或为空则返回空字典
	"""
	try:
		with open(path, 'r', encoding='utf-8') as f:
			return yaml.safe_load(f) or {}
	except FileNotFoundError:
		return {}


def deep_update(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
	"""
	深度合并两个字典，将b的键值对合并到a中

	Args:
		a: 目标字典（将被修改）
		b: 源字典（提供新值）

	Returns:
		合并后的字典a
	"""
	for k, v in b.items():
		if isinstance(v, dict) and isinstance(a.get(k), dict):
			a[k] = deep_update(a[k], v)
		else:
			a[k] = v
	return a


def parse_cli_overrides(overrides: List[str]) -> Dict[str, Any]:
	"""
	解析命令行覆盖参数，将key=value格式转换为嵌套字典

	使用点号表示嵌套路径，例如：
	["optimizer.lr=1e-3", "federated.num_clients=10"]

	Args:
		overrides: 命令行覆盖参数列表

	Returns:
		解析后的嵌套字典结构
	"""
	result: Dict[str, Any] = {}
	for item in overrides or []:
		if '=' not in item:
			continue
		key, val = item.split('=', 1)
		# 尝试转换数字和布尔值类型
		if val.lower() in {"true", "false"}:
			cast_val: Any = val.lower() == "true"
		else:
			try:
				if '.' in val:
					cast_val = float(val)
					if cast_val.is_integer():
						cast_val = int(cast_val)
				else:
					cast_val = int(val)
			except ValueError:
				# 如果无法转换为数字，保持原字符串格式
				cast_val = val
		nodes = key.split('.')
		cur = result
		for n in nodes[:-1]:
			if n not in cur or not isinstance(cur[n], dict):
				cur[n] = {}
			cur = cur[n]
		cur[nodes[-1]] = cast_val
	return result


def load_two_configs(arch_config: str, train_config: str, cli_overrides: List[str] = None) -> Dict[str, Any]:
	"""
	新的双配置加载器：合并架构配置、训练配置和命令行覆盖参数

	Args:
		arch_config: 架构/模型配置文件路径
		train_config: 训练/联邦超参数配置文件路径
		cli_overrides: 命令行覆盖参数列表

	Returns:
		合并后的完整配置字典
	"""
	cfg: Dict[str, Any] = {}
	cfg = deep_update(cfg, load_yaml(arch_config))
	cfg = deep_update(cfg, load_yaml(train_config))
	if cli_overrides:
		cfg = deep_update(cfg, parse_cli_overrides(cli_overrides))
	return cfg


def build_argparser() -> argparse.ArgumentParser:
	"""
	构建命令行参数解析器

	只支持双配置模式（arch-config + train-config）

	Returns:
		配置好的参数解析器对象
	"""
	parser = argparse.ArgumentParser()
	# 双配置模式参数（必需）
	parser.add_argument('--arch-config', type=str, required=True, help='architecture/model config yaml')
	parser.add_argument('--train-config', type=str, required=True, help='training/federated hyperparams yaml')
	parser.add_argument('--override', type=str, nargs='*', default=None, help='dot.notation overrides')
	return parser


def validate_training_config(cfg: Dict[str, Any]) -> None:
	"""
	验证训练配置的合法性，确保基模训练和LoRA微调的规则正确

	Args:
		cfg: 训练配置字典

	Raises:
		ValueError: 当配置不符合规则时抛出异常
	"""
	use_lora = cfg.get('use_lora', False)
	lora_cfg = cfg.get('lora', {})
	base_model_path = lora_cfg.get('base_model_path')

	print(f"[Config Validation] use_lora={use_lora}, base_model_path={base_model_path}")

	if use_lora:
		# LoRA微调模式的验证
		print("[Config Validation] LoRA fine-tuning mode detected")

		# 规则2: LoRA微调必须提供基模路径
		if not base_model_path or base_model_path == "null":
			raise ValueError(
				"❌ LoRA fine-tuning mode error: Must specify a valid base_model_path!\n"
				"Please set in federated.yaml:\n"
				"lora:\n"
				"  base_model_path: \"checkpoints/your_model/server/round_X.pth\""
			)

		# 规则1: 验证基模路径是否存在
		import os
		if not os.path.isabs(base_model_path):
			# 相对路径，基于outputs目录
			full_path = os.path.join(cfg.get('logging', {}).get('root', './outputs'), base_model_path)
		else:
			full_path = base_model_path

		if not os.path.exists(full_path):
			raise ValueError(
				f"❌ LoRA fine-tuning mode error: Base model path does not exist!\n"
				f"Specified path: {base_model_path}\n"
				f"Full path: {full_path}\n"
				f"Please ensure the base model file exists before LoRA fine-tuning."
			)

		print(f"[Config Validation] ✅ LoRA fine-tuning configuration is valid, base model path: {full_path}")

	else:
		# 基模训练模式的验证
		print("[Config Validation] Base model training mode detected")

		# 规则4: 基模训练时不应该有base_model_path
		if base_model_path and base_model_path != "null":
			raise ValueError(
				"❌ Base model training mode error: Should not set base_model_path!\n"
				"In base model training mode, please set:\n"
				"use_lora: false\n"
				"lora:\n"
				"  base_model_path: null"
			)

		print("[Config Validation] ✅ Base model training configuration is valid")

	print("[Config Validation] Training configuration validation passed")