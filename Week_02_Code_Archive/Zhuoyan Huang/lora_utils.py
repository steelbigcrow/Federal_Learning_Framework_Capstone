# lora_utils.py
"""
把已训练好的 RNNClassifier 就地转换为 LoRA 训练版本的实用函数：
- convert_to_lora_inplace: 将指定模块（默认 embedding 与 fc）替换为 loralib 模块，并拷贝基座权重
- mark_only_lora_as_trainable: 冻结除 LoRA 以外的参数（可选是否训练 bias）
- save_lora_adapters / load_lora_adapters: 仅保存/加载 LoRA 增量权重

使用流程（示例）：
    model = RNNClassifier(...)
    model.load_state_dict(torch.load('checkpoints/rnn_base.pt'))
    convert_to_lora_inplace(model, r=8, alpha=16, lora_dropout=0.05, target_modules=('embedding', 'fc'))
    mark_only_lora_as_trainable(model, train_bias=False)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-3, weight_decay=0.0)
"""

from typing import Iterable, Tuple
import warnings

import torch
import torch.nn as nn

try:
    import loralib as lora
except Exception as e:  # pragma: no cover
    raise ImportError(
        "Failed to import loralib. Please install it first, e.g. `pip install loralib`."
    ) from e


__all__ = [
    "convert_to_lora_inplace",
    "mark_only_lora_as_trainable",
    "save_lora_adapters",
    "load_lora_adapters",
]


def _to_new_module_device_dtype(new_module: nn.Module, ref_param: torch.Tensor) -> nn.Module:
    """
    将新模块移动到与参考参数相同的 device 和 dtype。
    """
    return new_module.to(device=ref_param.device, dtype=ref_param.dtype)


def convert_to_lora_inplace(
    model: nn.Module,
    r: int = 8,
    alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: Tuple[str, ...] = ("embedding", "fc"),
) -> nn.Module:
    """
    就地把模型中的指定模块替换为 LoRA 版本，并拷贝基座权重。
    目前支持：
      - nn.Embedding -> lora.Embedding
      - nn.Linear    -> lora.Linear

    参数
    - model: 你的 RNNClassifier 实例（或包含相同命名子模块的模型）
    - r, alpha, lora_dropout: LoRA 的超参数
    - target_modules: 需要替换为 LoRA 的模块名元组（默认 'embedding' 与 'fc'）

    返回
    - 原地修改过的 model（同时也返回以便链式调用）
    """
    for name in target_modules:
        if not hasattr(model, name):
            warnings.warn(f"[LoRA] target module `{name}` not found in model; skip.")
            continue

        old_mod = getattr(model, name)

        # Embedding -> LoRA Embedding
        if isinstance(old_mod, nn.Embedding) and not isinstance(old_mod, lora.Embedding):
            new_mod = lora.Embedding(
                num_embeddings=old_mod.num_embeddings,
                embedding_dim=old_mod.embedding_dim,
                padding_idx=old_mod.padding_idx,
                r=r,
                lora_alpha=alpha,
            )
            new_mod = _to_new_module_device_dtype(new_mod, old_mod.weight)
            with torch.no_grad():
                new_mod.weight.copy_(old_mod.weight)
            setattr(model, name, new_mod)

        # Linear -> LoRA Linear
        elif isinstance(old_mod, nn.Linear) and not isinstance(old_mod, lora.Linear):
            new_mod = lora.Linear(
                in_features=old_mod.in_features,
                out_features=old_mod.out_features,
                r=r,
                lora_alpha=alpha,
                lora_dropout=lora_dropout,
                bias=(old_mod.bias is not None),
            )
            # 依据权重 dtype/device 放置新模块
            new_mod = _to_new_module_device_dtype(new_mod, old_mod.weight)
            with torch.no_grad():
                new_mod.weight.copy_(old_mod.weight)
                if old_mod.bias is not None and new_mod.bias is not None:
                    new_mod.bias.copy_(old_mod.bias)
            setattr(model, name, new_mod)

        else:
            # 已经是 LoRA 模块或类型不支持
            if isinstance(old_mod, (lora.Linear, lora.Embedding)):
                warnings.warn(f"[LoRA] module `{name}` is already a LoRA module; skip.")
            else:
                warnings.warn(f"[LoRA] module `{name}` type {type(old_mod)} not supported; skip.")

    return model


def mark_only_lora_as_trainable(model: nn.Module, train_bias: bool = False) -> None:
    """
    冻结除 LoRA 参数外的所有参数。可选是否训练 bias。
    约定：LoRA 参数名中包含 'lora_'。
    """
    for n, p in model.named_parameters():
        if "lora_" in n:
            p.requires_grad = True
        elif n.endswith(".bias") and train_bias:
            p.requires_grad = True
        else:
            p.requires_grad = False


def save_lora_adapters(model: nn.Module, path: str) -> None:
    """
    仅保存 LoRA 相关增量权重（参数名包含 'lora_' 的条目）。
    """
    lora_sd = {k: v.detach().cpu() for k, v in model.state_dict().items() if "lora_" in k}
    if len(lora_sd) == 0:
        warnings.warn("[LoRA] No LoRA parameters found to save. Did you convert the model?")
    torch.save(lora_sd, path)


def load_lora_adapters(model: nn.Module, path: str, strict: bool = False, map_location: str = "cpu") -> None:
    """
    加载此前通过 save_lora_adapters 保存的增量权重。
    注意：调用前必须已将模型转换为 LoRA 版本（convert_to_lora_inplace）。
    """
    lora_sd = torch.load(path, map_location=map_location)
    missing_unexpected = model.load_state_dict(lora_sd, strict=strict)
    # 仅提示，不抛错，便于不同超参数/目标模块时的容错
    if missing_unexpected.missing_keys:
        warnings.warn(f"[LoRA] Missing keys when loading adapters: {missing_unexpected.missing_keys}")
    if missing_unexpected.unexpected_keys:
        warnings.warn(f"[LoRA] Unexpected keys when loading adapters: {missing_unexpected.unexpected_keys}")