# finetune_lora.py
import os
import argparse
import json
from datetime import datetime

import pandas as pd
import torch
import torch.nn as nn

from utils import set_seed, get_device, split_train_val, count_parameters, load_base_model_from_run
from data import create_datasets, build_dataloader
from engine import Trainer
from lora_utils import convert_to_lora_inplace, mark_only_lora_as_trainable, save_lora_adapters
from plot_utils import save_training_curves


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA finetuning on a pretrained RNN base model.")
    parser.add_argument("--csv", type=str, required=True, help="Path to csv for finetuning/eval.")
    parser.add_argument("--base_run_dir", type=str, required=True,
                        help="Path to pretrained RUN directory (contains rnn_base_*.pt, vocab.pkl, label_encoder.pkl, pretrain_config.json).")
    parser.add_argument("--output_dir", type=str, default="Checkpoints_lora", help="Dir to save LoRA adapters.")
    # Model arch (must match base)
    parser.add_argument("--embed_dim", type=int, default=300)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--max_len", type=int, default=200)
    # Training
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--val_batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3, help="Optimizer LR for LoRA params.")
    parser.add_argument("--max_lr", type=float, default=None, help="OneCycle max LR (default: same as --lr).")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--pct_start", type=float, default=0.1)
    parser.add_argument("--final_div_factor", type=float, default=10.0)
    parser.add_argument("--anneal_strategy", type=str, default="cos", choices=["cos", "linear"])
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--num_workers_train", type=int, default=8)
    parser.add_argument("--num_workers_eval", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    # LoRA
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", type=str, default="embedding,fc",
                        help="Comma separated module names to apply LoRA on, e.g., 'embedding,fc'")
    parser.add_argument("--train_bias", action="store_true", help="Whether to also train bias parameters.")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    # 统一一次训练产物的存储子目录（携带时间戳），便于归档与读取
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"rnn_lora_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    set_seed(args.seed, deterministic=True)
    device = get_device()
    print(f"Device: {device}")

    # 1) 从预训练运行目录加载：基座模型 + 词表 + 标签编码器 + 配置
    base_model, vocab, label_encoder, base_cfg = load_base_model_from_run(args.base_run_dir, map_location="cpu")

    # 2) 载入数据，并用相同的 label_encoder 转换标签
    df = pd.read_csv(args.csv).dropna()
    reviews = df["review"].tolist()
    labels_text = df["sentiment"].tolist()
    labels = label_encoder.transform(labels_text)  # 必须与预训练的类一致

    # 3) 划分数据集（可按需替换为你自己的划分方式）
    X_train, X_val, y_train, y_val = split_train_val(reviews, labels, test_size=0.3, seed=args.seed, stratify=True)

    # 4) 构建数据集/加载器（使用相同 vocab）
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

    # 5) 使用从 run 目录加载的基座模型（保证与预训练完全一致）
    model = base_model.to(device)

    # 6) 转换为 LoRA，并只训练 LoRA 参数（可选包含 bias）
    target_modules = tuple([s.strip() for s in args.target_modules.split(",") if s.strip()])
    convert_to_lora_inplace(
        model,
        r=args.lora_r,
        alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
    )
    mark_only_lora_as_trainable(model, train_bias=args.train_bias)

    # 简要打印参数规模
    total_params = count_parameters(model, only_trainable=False)
    trainable_params = count_parameters(model, only_trainable=True)
    print(f"Params: total={total_params:,} | trainable={trainable_params:,}")

    # 7) 组建优化器/调度器（仅 LoRA 参数）
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.99),
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    steps_per_epoch = len(train_loader)
    max_lr = args.max_lr if args.max_lr is not None else args.lr
    div_factor = max(max_lr / max(args.lr, 1e-12), 1.0)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        steps_per_epoch=steps_per_epoch,
        epochs=args.epochs,
        pct_start=args.pct_start,
        anneal_strategy=args.anneal_strategy,
        div_factor=div_factor,
        final_div_factor=args.final_div_factor,
    )

    # 8) 训练（微调）
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

    # 9) 仅保存 LoRA 适配器（文件名追加时间戳 + 子目录作为版本标识）
    adapters_path = os.path.join(run_dir, f"rnn_lora_adapters_{timestamp}.pt")
    save_lora_adapters(model, adapters_path)
    with open(os.path.join(run_dir, "finetune_lora_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    # 同步保存词表与标签编码器到本次 run 目录，便于独立分发
    try:
        import shutil
        for name in ["vocab.pkl", "label_encoder.pkl"]:
            src = os.path.join(args.base_run_dir, name)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(run_dir, name))
    except Exception as e:
        print(f"[Warn] Failed to copy artifacts to run dir: {e}")

    print(f"Saved LoRA adapters to: {adapters_path}")
    print(f"Saved artifacts to directory: {run_dir}/")

    # 10) 保存训练曲线与历史
    save_training_curves(history, out_dir=run_dir, tag=f"rnn_lora_{timestamp}")


if __name__ == "__main__":
    main()