# --- 兜底：确保 from src... 可导入 ---
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# ------------------------------------------------

import argparse, json, re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from sklearn.metrics import roc_auc_score, average_precision_score


# ---------------- 工具函数 ----------------
def _filter_state_to_model(model: nn.Module, sd: dict):
    """只保留与当前模型 shape 一致的键，避免 shape mismatch 报错"""
    kept, msd = {}, model.state_dict()
    for k, v in sd.items():
        if k in msd and msd[k].shape == v.shape:
            kept[k] = v
    return kept

def _maybe_resize_text_embedding(model: nn.Module, sd: dict):
    """
    若 checkpoint 的 embedding 词表大小与当前模型不同，则把当前模型 embedding resize 成 ckpt 的大小。
    仅在模型具有 `embedding: nn.Embedding` 且维度匹配时进行。
    """
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

def _load_full(model, ckpt_path: Path):
    """加载完整模型权重（非 LoRA），并自动做 embedding 尺寸对齐 + 过滤不匹配键"""
    obj = torch.load(ckpt_path, map_location="cpu")
    sd = obj.get("state_dict") if isinstance(obj, dict) else None
    if sd is None:
        sd = obj
    _maybe_resize_text_embedding(model, sd)
    kept = _filter_state_to_model(model, sd)
    model.load_state_dict(kept, strict=False)
    return model

def _read_sidecar_json(pth: Path):
    """读取与 ckpt 同名的 sidecar json（若存在），用于获取 dataset/model 配置"""
    js = pth.with_suffix(pth.suffix + ".json")
    if js.exists():
        try:
            return json.loads(js.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None

def _infer_from_run_dir(run_dir: Path):
    """从 run 目录名里猜测 dataset / model（例如 mnist_mlp_..., imdb_rnn_...）"""
    name = run_dir.name.lower()
    dataset = "imdb" if "imdb" in name else ("mnist" if "mnist" in name else None)
    model = None
    for m in ["rnn", "lstm", "transformer", "mlp", "vit"]:
        if m in name:
            model = m
            break
    return dataset, model

def build_model_for_ckpt(run_dir: Path, ckpt: Path):
    """
    构建模型：
    1) 优先从 sidecar json 里拿 dataset/model 配置；
    2) 拿不到则从 run 目录名推断；
    3) 用 create_model(dataset, model, cfg, extra) 实例化模型。
    """
    from src.models.registry import create_model

    js = _read_sidecar_json(ckpt)
    dataset = model = None
    model_cfg = {}
    if js:
        mi = js.get("model_info") or js
        if isinstance(mi, dict):
            dataset = (mi.get("dataset") or "").lower()
            model = (mi.get("model_type") or "").lower()
            model_cfg = mi.get("model_config") or {}
    if not dataset or not model:
        dataset, model = _infer_from_run_dir(run_dir)
    if not dataset or not model:
        raise SystemExit(f"无法从 {ckpt} 推断 dataset/model；请检查命名或 sidecar json。")

    extra = {}
    if dataset == "imdb":
        # 兜底：若没写就默认；真正权重会由 ckpt 覆盖
        extra = {
            "vocab_size": int(model_cfg.get("vocab_size", 30000)),
            "pad_idx": int(model_cfg.get("pad_idx", 1)),
        }
    model = create_model(dataset, model, model_cfg, extra=extra)
    return model, dataset


# ---------------- 数据加载 ----------------
def get_mnist_test_loader(bs=512):
    ds = load_dataset("mnist", split="test")
    import numpy as np

    def tfm(batch):
        imgs = np.stack([np.array(x, dtype="float32") / 255.0 for x in batch["image"]])[:, None, :, :]
        labels = np.array(batch["label"], dtype="int64")
        return {"x": torch.from_numpy(imgs), "y": torch.from_numpy(labels)}

    return DataLoader(ds.with_transform(tfm), batch_size=bs, shuffle=False, num_workers=0, pin_memory=True)

def get_imdb_test_loader(pad_idx=1, max_len=256, bs=128):
    ds = load_dataset("imdb", split="test")

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

    return DataLoader(ds.with_transform(tfm), batch_size=bs, shuffle=False, num_workers=0, pin_memory=True)


# ---------------- 评测 ----------------
@torch.no_grad()
def mnist_acc(model, loader, device):
    model.to(device).eval()
    n_total, n_correct = 0, 0
    for b in loader:
        y = b["y"].to(device)
        p = model(b["x"].to(device)).argmax(1)
        n_total += y.numel()
        n_correct += (p == y).sum().item()
    return n_correct / n_total

@torch.no_grad()
def imdb_auc(model, loader, device):
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
    return roc_auc_score(y_true, y_prob), average_precision_score(y_true, y_prob)


# ---------------- 主流程：多轮对比 ----------------
def main():
    ap = argparse.ArgumentParser("Compare metrics across rounds for a single run")
    ap.add_argument("--run-dir", required=True, help="例如 outputs/checkpoints/mnist_mlp_2025xxxx")
    ap.add_argument("--rounds", help="逗号分隔，如 1,2,3；缺省则自动找该 run 的所有 round_*.pth")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-root", default="outputs/viz")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    server_dir = run_dir if run_dir.name == "server" else (run_dir / "server")

    # 找到该 run 下所有 round_*.pth
    patt = re.compile(r"round_(\d+)\.pth$")
    found = sorted(
        [
            (int(m.group(1)), p)
            for p in server_dir.glob("round_*.pth")
            for m in [patt.fullmatch(p.name)]
            if m
        ],
        key=lambda x: x[0],
    )
    if not found:
        raise SystemExit(f"{server_dir} 下没有 round_*.pth")

    # 如用户指定 --rounds，则筛选
    if args.rounds:
        keep = {int(x) for x in args.rounds.split(",")}
        found = [t for t in found if t[0] in keep]
        if not found:
            raise SystemExit("指定的 --rounds 未找到相应的检查点")

    # 用第一轮构建模型和 DataLoader
    first_ckpt = found[0][1]
    model, dataset = build_model_for_ckpt(run_dir, first_ckpt)
    if dataset == "mnist":
        loader = get_mnist_test_loader()
    else:
        loader = get_imdb_test_loader()

    # 逐轮评测
    metrics = []
    for r, ck in found:
        _load_full(model, ck)
        if dataset == "mnist":
            acc = mnist_acc(model, loader, args.device)
            metrics.append((r, acc))
            print(f"[MNIST] round {r}: acc={acc:.4f}")
        else:
            roc_auc, pr_auc = imdb_auc(model, loader, args.device)
            metrics.append((r, roc_auc, pr_auc))
            print(f"[IMDB] round {r}: ROC-AUC={roc_auc:.4f}  PR-AUC={pr_auc:.4f}")

    # 输出目录
    outdir = Path(args.out_root) / run_dir.name / "compare_rounds"
    outdir.mkdir(parents=True, exist_ok=True)

    # 作图 + 保存 metrics.json
    if dataset == "mnist":
        rs = [r for r, _ in metrics]
        accs = [a for _, a in metrics]
        plt.figure()
        plt.plot(rs, accs, marker="o")
        plt.xlabel("Round")
        plt.ylabel("Accuracy")
        plt.title(run_dir.name)
        plt.grid(True, alpha=.3)
        plt.tight_layout()
        plt.savefig(outdir / "mnist_acc_vs_round.png", dpi=160)
        plt.close()

        (outdir / "metrics.json").write_text(
            json.dumps(
                {"dataset": "mnist", "by_round": [{"round": r, "accuracy": float(a)} for r, a in metrics]},
                indent=2, ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    else:  # imdb
        rs = [t[0] for t in metrics]
        roc = [t[1] for t in metrics]
        prc = [t[2] for t in metrics]
        plt.figure()
        plt.plot(rs, roc, marker="o", label="ROC-AUC")
        plt.plot(rs, prc, marker="s", label="PR-AUC")
        plt.xlabel("Round")
        plt.ylabel("AUC")
        plt.title(run_dir.name)
        plt.legend()
        plt.grid(True, alpha=.3)
        plt.tight_layout()
        plt.savefig(outdir / "imdb_auc_vs_round.png", dpi=160)
        plt.close()

        (outdir / "metrics.json").write_text(
            json.dumps(
                {"dataset": "imdb", "by_round": [{"round": r, "roc_auc": float(a), "pr_auc": float(b)} for r, a, b in metrics]},
                indent=2, ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    print(f"[compare] done → {outdir}")


if __name__ == "__main__":
    main()
