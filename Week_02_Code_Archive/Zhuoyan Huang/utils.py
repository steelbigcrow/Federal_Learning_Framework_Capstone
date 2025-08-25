# utils.py
from typing import Tuple, List, Any, Dict, Optional
from datetime import datetime
import json
import os
import random
import pickle

import numpy as np
import torch
from sklearn.model_selection import train_test_split


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model: torch.nn.Module, only_trainable: bool = False) -> int:
    params = model.parameters() if not only_trainable else (p for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in params)


def split_train_val(
    reviews: List[str],
    labels: List[int],
    test_size: float = 0.3,
    seed: int = 42,
    stratify: bool = True
) -> Tuple[List[str], List[str], List[int], List[int]]:
    strat = labels if stratify else None
    X_train, X_val, y_train, y_val = train_test_split(
        reviews, labels, test_size=test_size, random_state=seed, stratify=strat
    )
    return X_train, X_val, y_train, y_val


def save_pickle(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def format_timestamp(dt: datetime = None) -> str:
    """
    统一的时间戳格式化函数，默认当前时间。
    """
    dt = dt or datetime.now()
    return dt.strftime("%Y%m%d_%H%M%S")


def list_run_subdirs(parent_dir: str, prefix: str) -> List[str]:
    """
    列出 parent_dir 下，以 prefix 开头的子目录，并按时间戳倒序返回完整路径列表。
    例如：prefix='rnn_base_' 或 'rnn_lora_'
    """
    if not os.path.isdir(parent_dir):
        return []
    subdirs = []
    for name in os.listdir(parent_dir):
        full = os.path.join(parent_dir, name)
        if os.path.isdir(full) and name.startswith(prefix):
            subdirs.append(full)
    # 按名称中的时间戳倒序（若解析失败则保持原序）
    def _key(p: str) -> str:
        return p
    try:
        subdirs.sort(key=lambda p: os.path.basename(p).split(prefix, 1)[1], reverse=True)
    except Exception:
        subdirs.sort(reverse=True)
    return subdirs


def latest_run_subdir(parent_dir: str, prefix: str) -> str:
    """
    获取 parent_dir 下以 prefix 开头的最新一次 run 目录，若不存在返回空字符串。
    """
    subdirs = list_run_subdirs(parent_dir, prefix)
    return subdirs[0] if subdirs else ""


def load_base_model_from_run(run_dir: str, map_location: str = "cpu") -> Tuple[Any, Dict[str, int], Any, Dict[str, Any]]:
    """
    从一次基础预训练的 run 目录加载：模型、词表、标签编码器与配置。

    期望目录结构：
    - run_dir/
      - rnn_base_*.pt
      - vocab.pkl
      - label_encoder.pkl
      - pretrain_config.json
    """
    # 读取 artifacts 与配置
    vocab_path = os.path.join(run_dir, "vocab.pkl")
    le_path = os.path.join(run_dir, "label_encoder.pkl")
    cfg_path = os.path.join(run_dir, "pretrain_config.json")
    ckpt_path: Optional[str] = None
    for name in os.listdir(run_dir):
        if name.startswith("rnn_base_") and name.endswith(".pt"):
            ckpt_path = os.path.join(run_dir, name)
            break
    if ckpt_path is None:
        raise FileNotFoundError(f"No base checkpoint found in {run_dir}")

    vocab: Dict[str, int] = load_pickle(vocab_path)
    label_encoder = load_pickle(le_path)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg: Dict[str, Any] = json.load(f)

    # 构建模型并加载权重
    from models import RNNClassifier  # 延迟导入避免循环依赖
    pad_idx = vocab.get('<pad>', 1)
    model = RNNClassifier(
        vocab_size=len(vocab),
        embed_dim=cfg.get("embed_dim", 300),
        hidden_dim=cfg.get("hidden_dim", 512),
        output_dim=len(getattr(label_encoder, "classes_", [])) or 2,
        num_layers=cfg.get("layers", 4),
        pad_idx=pad_idx,
    )
    state_dict = torch.load(ckpt_path, map_location=map_location)
    model.load_state_dict(state_dict, strict=True)
    return model, vocab, label_encoder, cfg


def load_lora_model_from_run(run_dir: str, map_location: str = "cpu") -> Tuple[Any, Dict[str, int], Any, Dict[str, Any]]:
    """
    从一次 LoRA 微调的 run 目录加载：已应用 LoRA 的模型、词表、标签编码器与配置。

    期望目录结构：
    - run_dir/
      - rnn_lora_adapters_*.pt
      - vocab.pkl  (微调时已拷贝)
      - label_encoder.pkl (微调时已拷贝)
      - finetune_lora_config.json (包含 base_ckpt 与 target_modules 等)
    """
    adapters_ckpt: Optional[str] = None
    for name in os.listdir(run_dir):
        if name.startswith("rnn_lora_adapters_") and name.endswith(".pt"):
            adapters_ckpt = os.path.join(run_dir, name)
            break
    if adapters_ckpt is None:
        raise FileNotFoundError(f"No LoRA adapters checkpoint found in {run_dir}")

    vocab_path = os.path.join(run_dir, "vocab.pkl")
    le_path = os.path.join(run_dir, "label_encoder.pkl")
    cfg_path = os.path.join(run_dir, "finetune_lora_config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg: Dict[str, Any] = json.load(f)

    # 若 run_dir 未包含 artifacts，则优先回退到 base_run_dir，其次回退到 cfg.artifacts_dir，再次回退到 base_ckpt 所在目录
    if not os.path.isfile(vocab_path) or not os.path.isfile(le_path):
        fallback_dir = None
        if cfg.get("base_run_dir"):
            fallback_dir = cfg["base_run_dir"]
        elif cfg.get("artifacts_dir"):
            fallback_dir = cfg["artifacts_dir"]
        else:
            fallback_dir = os.path.dirname(os.path.abspath(cfg.get("base_ckpt", "")))
        vocab_path = os.path.join(fallback_dir, "vocab.pkl")
        le_path = os.path.join(fallback_dir, "label_encoder.pkl")

    vocab: Dict[str, int] = load_pickle(vocab_path)
    label_encoder = load_pickle(le_path)

    # 构建与 base 相同结构的模型并加载 base 权重
    from models import RNNClassifier
    from lora_utils import convert_to_lora_inplace, load_lora_adapters

    if cfg.get("base_run_dir"):
        base_model, base_vocab, base_le, base_cfg = load_base_model_from_run(cfg["base_run_dir"], map_location=map_location)
        # 使用从 run 目录加载的模型与 artifacts，覆盖上面从本 run 目录读取的 vocab/label
        model = base_model
        vocab = base_vocab
        label_encoder = base_le
    else:
        pad_idx = vocab.get('<pad>', 1)
        model = RNNClassifier(
            vocab_size=len(vocab),
            embed_dim=cfg.get("embed_dim", 300),
            hidden_dim=cfg.get("hidden_dim", 512),
            output_dim=len(getattr(label_encoder, "classes_", [])) or 2,
            num_layers=cfg.get("layers", 4),
            pad_idx=pad_idx,
        )
        base_sd = torch.load(cfg["base_ckpt"], map_location=map_location)
        model.load_state_dict(base_sd, strict=True)

    # 转换为 LoRA 并加载适配器
    target_modules = tuple(s.strip() for s in str(cfg.get("target_modules", "embedding,fc")).split(',') if s.strip())
    convert_to_lora_inplace(
        model,
        r=int(cfg.get("lora_r", 8)),
        alpha=int(cfg.get("lora_alpha", 16)),
        lora_dropout=float(cfg.get("lora_dropout", 0.05)),
        target_modules=target_modules,
    )
    load_lora_adapters(model, adapters_ckpt, strict=False, map_location=map_location)
    return model, vocab, label_encoder, cfg