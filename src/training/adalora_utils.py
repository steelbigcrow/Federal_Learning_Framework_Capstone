"""
AdaLoRA utilities for federated learning framework

This module provides AdaLoRA functionality using the implementation from:
https://github.com/QingruZhang/AdaLoRA

AdaLoRA (Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning) provides
dynamic rank allocation using SVD-based adaptation with automatic rank budget management.
"""

import os
import torch
import torch.nn as nn
from typing import Iterable, List, Dict, Optional
import sys

# Add the AdaLoRA library to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'adaloralib'))

try:
    from loralib.adalora import SVDLinear, RankAllocator, compute_orth_regu
    from loralib.utils import mark_only_lora_as_trainable as adalora_mark_trainable
    from loralib.utils import lora_state_dict as adalora_state_dict
except ImportError:
    # If loralib is not available, we'll provide fallback implementations
    print("[AdaLoRA] Warning: loralib not found. Please install from https://github.com/QingruZhang/AdaLoRA")
    SVDLinear = None
    RankAllocator = None
    compute_orth_regu = None


def _replace_modules_with_adalora(module: nn.Module, r: int, alpha: int, dropout: float, 
                                  target_modules: Iterable[str], prefix: str, replaced: List[str]) -> None:
    """
    递归替换模块为AdaLoRA版本
    
    Args:
        module: 要替换的模块
        r: 初始LoRA秩
        alpha: LoRA缩放因子
        dropout: dropout率
        target_modules: 目标模块类型列表
        prefix: 当前模块名前缀
        replaced: 已替换的模块列表
    """
    if SVDLinear is None:
        raise ImportError("AdaLoRA library not available. Please install from https://github.com/QingruZhang/AdaLoRA")
    
    for name, child in list(module.named_children()):
        full_name = f"{prefix}.{name}" if prefix else name
        cls_name = child.__class__.__name__
        
        # 替换Linear层
        if isinstance(child, nn.Linear) and (cls_name in target_modules or 'Linear' in target_modules):
            new_layer = SVDLinear(
                child.in_features, 
                child.out_features, 
                r=r, 
                lora_alpha=alpha, 
                lora_dropout=dropout,
                bias=child.bias is not None
            )
            # 复制权重
            new_layer.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                new_layer.bias.data.copy_(child.bias.data)
            setattr(module, name, new_layer)
            replaced.append(full_name)
        
        # 替换Embedding层（AdaLoRA主要针对Linear，Embedding保持标准LoRA）
        elif isinstance(child, nn.Embedding) and (cls_name in target_modules or 'Embedding' in target_modules):
            # 对于Embedding层，我们使用标准的LoRA实现
            try:
                from loralib import Embedding as LoRAEmbedding
                new_layer = LoRAEmbedding(
                    child.num_embeddings, 
                    child.embedding_dim, 
                    r=r, 
                    lora_alpha=alpha,
                    max_norm=child.max_norm, 
                    norm_type=child.norm_type, 
                    scale_grad_by_freq=child.scale_grad_by_freq, 
                    sparse=child.sparse, 
                    padding_idx=child.padding_idx
                )
                new_layer.weight.data.copy_(child.weight.data)
                setattr(module, name, new_layer)
                replaced.append(full_name)
            except ImportError:
                print(f"[AdaLoRA] Warning: Cannot replace Embedding layer {full_name} - loralib not available")
        
        else:
            _replace_modules_with_adalora(child, r, alpha, dropout, target_modules, full_name, replaced)


def inject_adalora_modules(model: nn.Module, r: int = 8, alpha: int = 16, dropout: float = 0.0, 
                          target_modules: Iterable[str] = ("Linear", "Embedding")) -> List[str]:
    """
    向模型中注入AdaLoRA模块
    
    Args:
        model: 要注入AdaLoRA的模型
        r: 初始LoRA秩
        alpha: LoRA缩放因子
        dropout: dropout率
        target_modules: 目标模块类型列表
    
    Returns:
        被替换的模块名称列表
    """
    if SVDLinear is None:
        raise ImportError("AdaLoRA library not available. Please install from https://github.com/QingruZhang/AdaLoRA")
    
    replaced: List[str] = []
    _replace_modules_with_adalora(model, r, alpha, dropout, target_modules, prefix="", replaced=replaced)
    return replaced


def mark_only_adalora_as_trainable(model: nn.Module, train_classifier_head: bool = True) -> None:
    """
    将模型设置为只训练AdaLoRA参数
    
    Args:
        model: 模型
        train_classifier_head: 是否同时训练分类器头部
    """
    if SVDLinear is None:
        raise ImportError("AdaLoRA library not available. Please install from https://github.com/QingruZhang/AdaLoRA")
    
    # 使用AdaLoRA的标记函数
    adalora_mark_trainable(model, bias='lora_only')
    
    # 如果需要，额外启用分类器头部
    if train_classifier_head:
        for name, module in model.named_modules():
            if name.endswith('classifier') or name.endswith('head'):
                for p in module.parameters():
                    p.requires_grad = True


def extract_adalora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    提取模型中所有AdaLoRA参数的state_dict
    
    Args:
        model: 包含AdaLoRA参数的模型
    
    Returns:
        包含AdaLoRA参数的字典
    """
    if SVDLinear is None:
        raise ImportError("AdaLoRA library not available. Please install from https://github.com/QingruZhang/AdaLoRA")
    
    # 使用AdaLoRA的状态字典提取函数
    return adalora_state_dict(model, bias='lora_only')


def load_adalora_state_dict(model: nn.Module, adalora_state_dict: Dict[str, torch.Tensor], strict: bool = False) -> None:
    """
    将AdaLoRA权重加载到模型中
    
    Args:
        model: 目标模型
        adalora_state_dict: AdaLoRA权重字典
        strict: 是否严格模式（缺少键时报错）
    """
    if SVDLinear is None:
        raise ImportError("AdaLoRA library not available. Please install from https://github.com/QingruZhang/AdaLoRA")
    
    model_dict = model.state_dict()
    
    # 只加载AdaLoRA相关的参数
    for key, value in adalora_state_dict.items():
        if key in model_dict:
            try:
                model_dict[key].data.copy_(value.data)
            except Exception as e:
                if strict:
                    raise ValueError(f"Failed to load parameter {key}: {e}")
                else:
                    print(f"[AdaLoRA] Warning: Failed to load parameter {key}: {e}")
        elif strict:
            raise KeyError(f"Missing AdaLoRA key: {key}")


def create_rank_allocator(model: nn.Module, config: Dict) -> Optional[RankAllocator]:
    """
    创建AdaLoRA的RankAllocator
    
    Args:
        model: 要管理的模型
        config: AdaLoRA配置字典
    
    Returns:
        RankAllocator实例或None
    """
    if RankAllocator is None:
        raise ImportError("AdaLoRA library not available. Please install from https://github.com/QingruZhang/AdaLoRA")
    
    return RankAllocator(
        model=model,
        lora_r=config.get('initial_r', 8),
        target_rank=config.get('target_rank', 4),
        init_warmup=config.get('init_warmup', 125),
        final_warmup=config.get('final_warmup', 375),
        mask_interval=config.get('mask_interval', 10),
        beta1=config.get('beta1', 0.85),
        beta2=config.get('beta2', 0.85),
        total_step=config.get('total_step', None),
        target_total_rank=config.get('total_rank', None)
    )


def save_adalora_checkpoint(model: nn.Module, path: str, metadata: Dict = None) -> None:
    """
    保存AdaLoRA权重检查点
    
    Args:
        model: 包含AdaLoRA参数的模型
        path: 保存路径
        metadata: 元数据信息
    """
    if SVDLinear is None:
        raise ImportError("AdaLoRA library not available. Please install from https://github.com/QingruZhang/AdaLoRA")
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    adalora_state_dict = extract_adalora_state_dict(model)
    checkpoint = {
        'adalora_state_dict': adalora_state_dict,
        'metadata': metadata or {}
    }
    
    torch.save(checkpoint, path)


def load_adalora_checkpoint(model: nn.Module, path: str, strict: bool = False) -> Dict:
    """
    加载AdaLoRA权重检查点
    
    Args:
        model: 目标模型
        path: 检查点文件路径
        strict: 是否严格模式
    
    Returns:
        检查点中的元数据
    """
    if SVDLinear is None:
        raise ImportError("AdaLoRA library not available. Please install from https://github.com/QingruZhang/AdaLoRA")
    
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    
    if 'adalora_state_dict' in checkpoint:
        load_adalora_state_dict(model, checkpoint['adalora_state_dict'], strict=strict)
    else:
        # 兼容旧格式
        load_adalora_state_dict(model, checkpoint, strict=strict)
    
    return checkpoint.get('metadata', {})


def get_adalora_regularization_loss(model: nn.Module, weight: float = 0.1) -> torch.Tensor:
    """
    计算AdaLoRA的正交正则化损失
    
    Args:
        model: 模型
        weight: 正则化权重
    
    Returns:
        正交正则化损失
    """
    if compute_orth_regu is None:
        raise ImportError("AdaLoRA library not available. Please install from https://github.com/QingruZhang/AdaLoRA")
    
    return compute_orth_regu(model, regu_weight=weight)


def get_adalora_parameter_stats(model: nn.Module) -> Dict[str, int]:
    """
    获取AdaLoRA模型的参数统计信息
    
    Args:
        model: AdaLoRA模型
    
    Returns:
        参数统计信息字典
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 统计AdaLoRA特有的参数
    adalora_params = 0
    rank_params = 0
    
    for name, param in model.named_parameters():
        if 'lora_' in name:
            adalora_params += param.numel()
        if 'ranknum' in name:
            rank_params += param.numel()
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'adalora_params': adalora_params,
        'rank_params': rank_params,
        'trainable_percentage': 100 * trainable_params / total_params if total_params > 0 else 0
    }