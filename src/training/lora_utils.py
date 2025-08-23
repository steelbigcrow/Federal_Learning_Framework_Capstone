from typing import Iterable, List, Dict
import torch
import torch.nn as nn
import loralib as lora
import os


def _replace_modules_with_lora(module: nn.Module, r: int, alpha: int, dropout: float, target_modules: Iterable[str], prefix: str, replaced: List[str]) -> None:
	for name, child in list(module.named_children()):
		full_name = f"{prefix}.{name}" if prefix else name
		cls_name = child.__class__.__name__
		
		# 替换Linear层
		if isinstance(child, nn.Linear) and (cls_name in target_modules or 'Linear' in target_modules):
			new_layer = lora.Linear(child.in_features, child.out_features, r=r, lora_alpha=alpha, lora_dropout=dropout, bias=child.bias is not None)
			new_layer.weight.data.copy_(child.weight.data)
			if child.bias is not None:
				new_layer.bias.data.copy_(child.bias.data)
			setattr(module, name, new_layer)
			replaced.append(full_name)
		
		# 替换Embedding层
		elif isinstance(child, nn.Embedding) and (cls_name in target_modules or 'Embedding' in target_modules):
			new_layer = lora.Embedding(child.num_embeddings, child.embedding_dim, r=r, lora_alpha=alpha, max_norm=child.max_norm, norm_type=child.norm_type, scale_grad_by_freq=child.scale_grad_by_freq, sparse=child.sparse, padding_idx=child.padding_idx)
			new_layer.weight.data.copy_(child.weight.data)
			setattr(module, name, new_layer)
			replaced.append(full_name)
		
		else:
			_replace_modules_with_lora(child, r, alpha, dropout, target_modules, full_name, replaced)


def inject_lora_modules(model: nn.Module, r: int = 8, alpha: int = 16, dropout: float = 0.0, target_modules: Iterable[str] = ("Linear", "Embedding")) -> List[str]:
	"""
	注入LoRA到指定的模块类型中（Linear和Embedding）

	Args:
		model: 要注入LoRA的模型
		r: LoRA的秩
		alpha: LoRA的缩放因子
		dropout: LoRA的dropout率
		target_modules: 目标模块类型列表

	Returns:
		被替换的模块名称列表
	"""
	replaced: List[str] = []
	_replace_modules_with_lora(model, r, alpha, dropout, target_modules, prefix="", replaced=replaced)
	return replaced



def mark_only_lora_as_trainable(model: nn.Module, train_classifier_head: bool = True) -> None:
	"""
	将模型设置为只训练LoRA参数

	Args:
		model: 模型
		train_classifier_head: 是否同时训练分类器头部
	"""
	lora.mark_only_lora_as_trainable(model)
	if train_classifier_head:
		for name, module in model.named_modules():
			if name.endswith('classifier') or name.endswith('head'):
				for p in module.parameters():
					p.requires_grad = True


def extract_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
	"""
	提取模型中所有LoRA参数的state_dict

	Args:
		model: 包含LoRA参数的模型

	Returns:
		包含LoRA参数的字典
	"""
	lora_state_dict = {}
	for name, module in model.named_modules():
		if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
			# LoRA层
			lora_state_dict[f"{name}.lora_A"] = module.lora_A.detach().clone()
			lora_state_dict[f"{name}.lora_B"] = module.lora_B.detach().clone()
			if hasattr(module, 'lora_embedding_A'):
				lora_state_dict[f"{name}.lora_embedding_A"] = module.lora_embedding_A.detach().clone()
			if hasattr(module, 'lora_embedding_B'):
				lora_state_dict[f"{name}.lora_embedding_B"] = module.lora_embedding_B.detach().clone()
	return lora_state_dict


def load_lora_state_dict(model: nn.Module, lora_state_dict: Dict[str, torch.Tensor], strict: bool = False) -> None:
	"""
	将LoRA权重加载到模型中

	Args:
		model: 目标模型
		lora_state_dict: LoRA权重字典
		strict: 是否严格模式（缺少键时报错）
	"""
	model_dict = {}
	for name, module in model.named_modules():
		if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
			# LoRA层
			lora_a_key = f"{name}.lora_A"
			lora_b_key = f"{name}.lora_B"

			if lora_a_key in lora_state_dict:
				module.lora_A.data.copy_(lora_state_dict[lora_a_key])
			elif strict:
				raise KeyError(f"Missing LoRA key: {lora_a_key}")

			if lora_b_key in lora_state_dict:
				module.lora_B.data.copy_(lora_state_dict[lora_b_key])
			elif strict:
				raise KeyError(f"Missing LoRA key: {lora_b_key}")

			# 处理embedding LoRA（如果存在）
			if hasattr(module, 'lora_embedding_A'):
				lora_emb_a_key = f"{name}.lora_embedding_A"
				if lora_emb_a_key in lora_state_dict:
					module.lora_embedding_A.data.copy_(lora_state_dict[lora_emb_a_key])

			if hasattr(module, 'lora_embedding_B'):
				lora_emb_b_key = f"{name}.lora_embedding_B"
				if lora_emb_b_key in lora_state_dict:
					module.lora_embedding_B.data.copy_(lora_state_dict[lora_emb_b_key])


def save_lora_checkpoint(model: nn.Module, path: str, metadata: Dict = None) -> None:
	"""
	保存LoRA权重检查点

	Args:
		model: 包含LoRA参数的模型
		path: 保存路径
		metadata: 元数据信息
	"""
	os.makedirs(os.path.dirname(path), exist_ok=True)

	lora_state_dict = extract_lora_state_dict(model)
	checkpoint = {
		'lora_state_dict': lora_state_dict,
		'metadata': metadata or {}
	}

	torch.save(checkpoint, path)


def load_lora_checkpoint(model: nn.Module, path: str, strict: bool = False) -> Dict:
	"""
	加载LoRA权重检查点

	Args:
		model: 目标模型
		path: 检查点文件路径
		strict: 是否严格模式

	Returns:
		检查点中的元数据
	"""
	checkpoint = torch.load(path, map_location='cpu')

	if 'lora_state_dict' in checkpoint:
		load_lora_state_dict(model, checkpoint['lora_state_dict'], strict=strict)
	else:
		# 兼容旧格式
		load_lora_state_dict(model, checkpoint, strict=strict)

	return checkpoint.get('metadata', {})


def load_base_model_checkpoint(model: nn.Module, checkpoint_path: str, strict: bool = True) -> Dict:
	"""
	从检查点加载基模权重

	Args:
		model: 目标模型
		checkpoint_path: 检查点文件路径
		strict: 是否严格模式

	Returns:
		检查点中的元数据信息
	"""
	if not os.path.exists(checkpoint_path):
		raise FileNotFoundError(f"Base model checkpoint not found: {checkpoint_path}")

	print(f"[LoRA] Loading base model: {checkpoint_path}")
	checkpoint = torch.load(checkpoint_path, map_location='cpu')

	# 处理不同的检查点格式
	if 'model_state_dict' in checkpoint:
		state_dict = checkpoint['model_state_dict']
		metadata = {k: v for k, v in checkpoint.items() if k not in ['model_state_dict']}
	elif 'state_dict' in checkpoint:
		state_dict = checkpoint['state_dict']
		metadata = {k: v for k, v in checkpoint.items() if k not in ['state_dict']}
	else:
		state_dict = checkpoint
		metadata = {}

	model.load_state_dict(state_dict, strict=strict)
	print(f"[LoRA] Base model loading completed")
	return metadata
