# plot_utils.py
from typing import List, Dict, Any
import os
import csv


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _save_history_csv(history: List[Dict[str, Any]], out_path: str) -> None:
    if not history:
        return
    keys = sorted(set().union(*(h.keys() for h in history)))
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def save_training_curves(history: List[Dict[str, Any]], out_dir: str, tag: str = "run") -> None:
    """
    根据 Trainer.fit 返回的 history（每 epoch 的指标字典）保存曲线图与 CSV：
    - {tag}_loss.png:     train_loss / val_loss 曲线
    - {tag}_acc.png:      train_acc  / val_acc  曲线（若存在）
    - {tag}_history.csv:  完整的逐 epoch 指标
    若未安装 matplotlib，将仅保存 CSV。
    """
    _ensure_dir(out_dir)

    csv_path = os.path.join(out_dir, f"{tag}_history.csv")
    _save_history_csv(history, csv_path)

    # 尝试导入 matplotlib；失败则只保存 CSV
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        print(f"[Warn] matplotlib not available. Curves not plotted. History saved to: {csv_path}")
        return

    if not history:
        print("[Warn] Empty history. Nothing to plot.")
        return

    epochs = [h.get("epoch", i + 1) for i, h in enumerate(history)]
    train_loss = [h.get("train_loss") for h in history]
    val_loss = [h.get("val_loss") for h in history if "val_loss" in h]
    has_val = len(val_loss) == len(history)

    # Loss figure
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, train_loss, label="train_loss")
    if has_val:
        plt.plot(epochs, [h.get("val_loss") for h in history], label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(f"Loss Curves ({tag})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    loss_path = os.path.join(out_dir, f"{tag}_loss.png")
    plt.savefig(loss_path)
    plt.close()

    # Accuracy figure (if available)
    if any("train_acc" in h for h in history):
        plt.figure(figsize=(7, 4))
        plt.plot(epochs, [h.get("train_acc") for h in history], label="train_acc")
        if has_val and any("val_acc" in h for h in history):
            plt.plot(epochs, [h.get("val_acc") for h in history], label="val_acc")
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.title(f"Accuracy Curves ({tag})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        acc_path = os.path.join(out_dir, f"{tag}_acc.png")
        plt.savefig(acc_path)
        plt.close()

