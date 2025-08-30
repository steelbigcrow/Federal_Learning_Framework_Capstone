# pretrain.py
import os
import argparse
import json
from datetime import datetime

import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder

from utils import set_seed, get_device, save_pickle, split_train_val
from data import build_vocab, create_datasets, build_dataloader
from models import RNNClassifier
from engine import Trainer
from plot_utils import save_training_curves


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain vanilla RNN classifier (base model).")
    parser.add_argument("--csv", type=str, required=True, help="Path to IMDB csv (with 'review' and 'sentiment').")
    parser.add_argument("--output_dir", type=str, default="Checkpoints", help="Directory to save artifacts.")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--val_batch_size", type=int, default=2048)
    parser.add_argument("--embed_dim", type=int, default=300)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4, help="Base LR for optimizer (and OneCycle initial LR).")
    parser.add_argument("--max_lr", type=float, default=6e-4, help="OneCycle max LR.")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--max_len", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers_train", type=int, default=8)
    parser.add_argument("--num_workers_eval", type=int, default=16)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--use_amp", action="store_true", help="Enable mixed precision training (AMP).")
    parser.add_argument("--pct_start", type=float, default=0.1, help="OneCycle pct_start.")
    parser.add_argument("--final_div_factor", type=float, default=10.0, help="OneCycle final_div_factor.")
    parser.add_argument("--anneal_strategy", type=str, default="cos", choices=["cos", "linear"])
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    # 统一一次训练产物的存储子目录（携带时间戳），便于归档与读取
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"rnn_base_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    set_seed(args.seed, deterministic=True)
    device = get_device()
    print(f"Device: {device}")

    # 1) 读取 CSV，并编码标签
    df = pd.read_csv(args.csv).dropna()
    reviews = df["review"].tolist()
    labels_text = df["sentiment"].tolist()

    le = LabelEncoder()
    labels = le.fit_transform(labels_text)  # e.g., positive->1, negative->0

    # 2) 划分数据集
    X_train, X_val, y_train, y_val = split_train_val(reviews, labels, test_size=0.3, seed=args.seed, stratify=True)

    # 3) 构建词表（基于训练集），保存
    vocab = build_vocab(X_train, min_freq=3, max_vocab_size=30000)
    save_pickle(vocab, os.path.join(run_dir, "vocab.pkl"))
    save_pickle(le, os.path.join(run_dir, "label_encoder.pkl"))

    # 4) 构建数据集和 DataLoader
    train_ds, val_ds, pad_idx, _ = create_datasets(X_train, y_train, X_val, y_val, vocab, max_len=args.max_len)
    train_loader = build_dataloader(
        dataset=train_ds,
        batch_size=args.batch_size,
        pad_idx=pad_idx,
        shuffle=True,
        num_workers=args.num_workers_train,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = build_dataloader(
        dataset=val_ds,
        batch_size=args.val_batch_size,
        pad_idx=pad_idx,
        shuffle=False,
        num_workers=args.num_workers_eval,
        pin_memory=True,
        persistent_workers=True,
    )

    # 5) 构建模型、损失、优化器、调度器
    model = RNNClassifier(
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        output_dim=len(le.classes_),
        num_layers=args.layers,
        pad_idx=pad_idx,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.99))

    steps_per_epoch = len(train_loader)
    # 设置 OneCycle 初始/最大/最终 LR
    div_factor = max(args.max_lr / max(args.lr, 1e-12), 1.0)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.max_lr,
        steps_per_epoch=steps_per_epoch,
        epochs=args.epochs,
        pct_start=args.pct_start,
        anneal_strategy=args.anneal_strategy,
        div_factor=div_factor,
        final_div_factor=args.final_div_factor,
    )

    # 6) 训练
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler,
        scheduler_step_per_batch=True,
        grad_clip=args.grad_clip,
        use_amp=args.use_amp,
    )
    history = trainer.fit(train_loader, val_loader=val_loader, num_epochs=args.epochs, log_interval=None)

    # 7) 保存基座模型与配置（文件名追加时间戳 + 子目录作为版本标识）
    base_ckpt = os.path.join(run_dir, f"rnn_base_{timestamp}.pt")
    torch.save(model.state_dict(), base_ckpt)
    with open(os.path.join(run_dir, "pretrain_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    # 8) 保存训练曲线与历史
    save_training_curves(history, out_dir=run_dir, tag=f"rnn_base_{timestamp}")

    print(f"Saved base model to: {base_ckpt}")
    print(f"Saved artifacts to directory: {run_dir}/")


if __name__ == "__main__":
    main()