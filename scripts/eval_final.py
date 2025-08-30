# --- 兜底：确保 from src... 可导入 ---
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# ------------------------------------------------

import argparse, json, math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, precision_recall_curve,
    roc_auc_score, average_precision_score
)

try:
    import yaml
except Exception:
    yaml = None


# ---------------- 工具函数 ----------------
def _read_yaml(p: Path) -> dict:
    if not yaml: return {}
    try:
        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}

def _read_sidecar_json(pth: Path):
    js = pth.with_suffix(pth.suffix + ".json")
    if js.exists():
        try:
            return json.loads(js.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None

def _filter_state_to_model(model: nn.Module, sd: dict):
    kept, msd = {}, model.state_dict()
    for k, v in sd.items():
        if k in msd and msd[k].shape == v.shape:
            kept[k] = v
    return kept

def _maybe_resize_text_embedding(model: nn.Module, sd: dict):
    # 如果 checkpoint 的 embedding 词表大小与当前模型不同，则把当前模型 embedding resize 成 checkpoint 的大小
    key = None
    for k in sd.keys():
        if k.endswith("embedding.weight"):
            key = k
            break
    if key is None:
        return
    if not hasattr(model, "embedding") or not isinstance(model.embedding, nn.Embedding):
        return
    ck_vocab, ck_dim = sd[key].shape
    cur_vocab, cur_dim = model.embedding.weight.shape
    if ck_dim != cur_dim:
        return
    if ck_vocab != cur_vocab:
        pad_idx = getattr(model, "pad_idx", model.embedding.padding_idx)
        new_emb = nn.Embedding(ck_vocab, ck_dim, padding_idx=pad_idx)
        new_emb = new_emb.to(model.embedding.weight.device, dtype=model.embedding.weight.dtype)
        model.embedding = new_emb

def _load_full(model, ck: Path):
    obj = torch.load(ck, map_location="cpu")
    sd = obj.get("state_dict") if isinstance(obj, dict) else None
    if sd is None:
        sd = obj
    _maybe_resize_text_embedding(model, sd)
    kept = _filter_state_to_model(model, sd)
    missed = [(k, tuple(sd[k].shape), tuple(model.state_dict()[k].shape))
              for k in sd.keys() if k not in kept and k in model.state_dict()]
    if missed:
        print(f"[eval] 警告：{len(missed)} 个参数形状不匹配，已跳过（示例）:", missed[:1])
    model.load_state_dict(kept, strict=False)
    return model

def _load_lora(model, base_ckpt: Path, lora_ckpt: Path):
    _load_full(model, base_ckpt)  # 先加载基模
    obj = torch.load(lora_ckpt, map_location="cpu")
    lsd = obj.get("state_dict") if isinstance(obj, dict) else None
    if lsd is None:
        lsd = obj
    kept = _filter_state_to_model(model, lsd)
    model.load_state_dict(kept, strict=False)
    return model


# ---------------- 构建模型 ----------------
def build_model(arch_cfg_path: Path):
    """根据 arch yaml 构建模型（走你项目里的 registry）"""
    from src.models.registry import create_model

    cfg = _read_yaml(arch_cfg_path)
    dataset = (cfg.get("dataset") or "").lower()
    model_name = (cfg.get("model") or "").lower()
    if not dataset or not model_name:
        raise SystemExit(f"arch 配置缺少 dataset/model: {arch_cfg_path}")

    extra = {}
    if dataset == "imdb":
        # 若 YAML 没写就给默认；真正的权重会被 checkpoint 覆盖
        extra = {"vocab_size": int(cfg.get("vocab_size", 30000)),
                 "pad_idx": int(cfg.get("pad_idx", 1))}
    model = create_model(dataset, model_name, cfg, extra=extra)
    return model, dataset


# ---------------- 数据加载 ----------------
def get_mnist_test_loader(batch_size=512):
    ds = load_dataset("mnist", split="test")
    import numpy as np

    def tfm(batch):
        imgs = np.stack([np.array(x, dtype="float32") / 255.0 for x in batch["image"]])[:, None, :, :]
        labels = np.array(batch["label"], dtype="int64")
        return {"x": torch.from_numpy(imgs), "y": torch.from_numpy(labels)}

    return DataLoader(ds.with_transform(tfm), batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

def get_imdb_test_loader(pad_idx=1, max_len=256, batch_size=128):
    ds = load_dataset("imdb", split="test")

    # 简单 tokenizer；如果你项目里有更好的 tokenizer，也可在这里替换
    def tok(text):
        ids = [hash(w) % 30000 for w in text.split()][:max_len]
        if len(ids) < max_len:
            ids += [pad_idx] * (max_len - len(ids))
        return ids

    import numpy as np

    def tfm(batch):
        X = np.stack([np.array(tok(t), dtype="int64") for t in batch["text"]])
        y = np.array(batch["label"], dtype="int64")
        return {"x": torch.from_numpy(X), "y": torch.from_numpy(y)}

    return DataLoader(ds.with_transform(tfm), batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)


# ---------------- 评测与画图 ----------------
@torch.no_grad()
def eval_mnist(model, loader, device="cuda"):
    model.to(device).eval()
    ys, ps = [], []
    for b in loader:
        logits = model(b["x"].to(device))
        ps.append(logits.argmax(1).cpu())
        ys.append(b["y"].cpu())
    y = torch.cat(ys).numpy()
    p = torch.cat(ps).numpy()
    cm = confusion_matrix(y, p, labels=list(range(10)))
    acc = (y == p).mean()
    return acc, cm, y, p

def plot_mnist(outdir: Path, acc, cm, y, p, mis_limit=36):
    outdir.mkdir(parents=True, exist_ok=True)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(10)))
    disp.plot(values_format="d")
    plt.title(f"MNIST Confusion Matrix (acc={acc:.4f})")
    plt.tight_layout()
    plt.savefig(outdir / "mnist_confusion_matrix.png", dpi=160)
    plt.close()

    per_cls = cm.diagonal() / cm.sum(1).clip(min=1)
    plt.figure()
    plt.bar(range(10), per_cls)
    plt.ylim(0, 1)
    plt.xlabel("Class")
    plt.ylabel("Accuracy")
    plt.title("Per-Class Accuracy")
    plt.tight_layout()
    plt.savefig(outdir / "mnist_per_class_acc.png", dpi=160)
    plt.close()

    # 误分类样例（注意索引转 int）
    import numpy as np
    mis = np.where(y != p)[0].astype(int)[:mis_limit]
    if len(mis) > 0:
        ds = load_dataset("mnist", split="test")
        cols = 6
        rows = math.ceil(len(mis) / cols)
        plt.figure(figsize=(cols * 2, rows * 2))
        for i, idx in enumerate(mis):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(ds[int(idx)]["image"], cmap="gray")
            plt.axis("off")
            plt.title(f"gt:{int(y[idx])} pred:{int(p[idx])}")
        plt.tight_layout()
        plt.savefig(outdir / "mnist_misclassified.png", dpi=160)
        plt.close()

@torch.no_grad()
def eval_imdb(model, loader, device="cuda"):
    model.to(device).eval()
    y_true, y_prob = [], []
    for b in loader:
        logits = model(b["x"].to(device))
        prob1 = torch.softmax(logits, 1)[:, 1].cpu().numpy()
        y_true.extend(b["y"].numpy())
        y_prob.extend(prob1.tolist())
    import numpy as np
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    return roc_auc, pr_auc, (fpr, tpr), (prec, rec)

def plot_imdb(outdir: Path, roc_auc, pr_auc, roc, pr):
    outdir.mkdir(parents=True, exist_ok=True)
    fpr, tpr = roc
    prec, rec = pr
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"ROC (AUC={roc_auc:.4f})")
    plt.tight_layout()
    plt.savefig(outdir / "imdb_roc.png", dpi=160)
    plt.close()

    plt.figure()
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR (AUC={pr_auc:.4f})")
    plt.tight_layout()
    plt.savefig(outdir / "imdb_pr.png", dpi=160)
    plt.close()


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser("Evaluate ONE federated global checkpoint")
    ap.add_argument("--arch-config", required=True)
    ap.add_argument("--checkpoint", required=True, help="server/round_N.pth (LoRA时为基模)")
    ap.add_argument("--lora-ckpt", help="LoRA 适配器（可选）")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--mis-limit", type=int, default=36)
    args = ap.parse_args()

    arch_path = Path(args.arch_config)
    model, dataset = build_model(arch_path)

    # 如果 sidecar 里带有 pad_idx / vocab_size，补齐到模型构造配置（仅影响数据管道）
    side = _read_sidecar_json(Path(args.checkpoint))
    if side and dataset == "imdb":
        mi = side.get("model_info") or {}
        mc = (mi.get("model_config") or {})
        pad_idx = int(mc.get("pad_idx", 1))
        vocab_size = int(mc.get("vocab_size", 30000))
    else:
        # 从 YAML 兜底，没有就默认
        cfg = _read_yaml(arch_path)
        pad_idx = int(cfg.get("pad_idx", 1))
        vocab_size = int(cfg.get("vocab_size", 30000))

    # 加载权重
    if args.lora_ckpt:
        model = _load_lora(model, Path(args.checkpoint), Path(args.lora_ckpt))
    else:
        model = _load_full(model, Path(args.checkpoint))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if dataset == "mnist":
        loader = get_mnist_test_loader()
        acc, cm, y, p = eval_mnist(model, loader, device=args.device)
        plot_mnist(outdir, acc, cm, y, p, mis_limit=args.mis_limit)
        print(f"[MNIST] acc={acc:.4f}  → {outdir}")
    else:
        loader = get_imdb_test_loader(pad_idx=pad_idx)
        roc_auc, pr_auc, roc, pr = eval_imdb(model, loader, device=args.device)
        plot_imdb(outdir, roc_auc, pr_auc, roc, pr)
        print(f"[IMDB] ROC-AUC={roc_auc:.4f} PR-AUC={pr_auc:.4f}  → {outdir}")


if __name__ == "__main__":
    main()
