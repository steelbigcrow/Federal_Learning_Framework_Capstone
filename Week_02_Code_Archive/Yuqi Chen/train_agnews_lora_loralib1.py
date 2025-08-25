"""
LoRA + DistilBERT on AG News using loralib 
- 仅替换注意力中的线性层为 loralib.Linear
- 冻结底模，只训练 LoRA 参数 + 分类头
- 支持命令行调整 LoRA 超参（r/alpha/dropout/target 模块）
- 记录运行配置、训练速度与最终指标到 JSON
- 每个 epoch 自动评估（用自定义回调），从训练集中切出验证集

运行示例：
  # 全参微调（Baseline）
  python train_agnews_lora_loralib1.py --epochs 1 --max_length 128 \
      --output_dir ./results_full_e1 --save_dir ./saved_full_e1 --lr 2e-5

  # LoRA (r=8, q/v/out) 
  python train_agnews_lora_loralib1.py --use_lora \
      --lora_r 8 --lora_alpha 16 --lora_targets q_lin,v_lin,out_lin \
      --epochs 1 --max_length 128 --lr 5e-5 \
      --output_dir ./results_lora_r8_e1 --save_dir ./saved_lora_r8_e1
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  

import json
import argparse
import random
import time
import math
import numpy as np
import torch
import torch.nn as nn

import loralib as lora  

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)

# =========================
# Utils
# =========================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        print("Using Apple MPS (Metal) for training.")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("Using NVIDIA CUDA for training.")
        return torch.device("cuda")
    else:
        print("No GPU backend detected. Using CPU (slower).")
        return torch.device("cpu")

def count_trainable_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# —— 每个 epoch 结束自动评估（兼容旧版，不用 evaluation_strategy）
class EvalAtEpochEnd(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        control.should_evaluate = True
        return control

# =========================
# 数据集装载
# =========================
def load_dataset_by_name(name: str):
    if name == "ag_news":
        ds = load_dataset("ag_news")  # {'train','test'}
        num_labels = 4
        text_key = "text"
        label_key = "label"
        return ds, num_labels, text_key, label_key
    else:
        raise ValueError(f"Unsupported dataset: {name}")

# =========================
# LoRA 注入（使用 loralib）
# =========================
def _replace_linear_with_lora(parent: nn.Module, child_name: str,
                              r: int, alpha: int, dropout: float):
    """
    将 parent.{child_name} (必须是 nn.Linear) 替换为 loralib.Linear
    并拷贝原权重/偏置，LoRA 低秩分支随机初始化。
    """
    old: nn.Linear = getattr(parent, child_name)
    if not isinstance(old, nn.Linear):
        return False

    new = lora.Linear(
        in_features=old.in_features,
        out_features=old.out_features,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias=(old.bias is not None),
    )
    with torch.no_grad():
        new.weight.copy_(old.weight)
        if old.bias is not None and new.bias is not None:
            new.bias.copy_(old.bias)

    setattr(parent, child_name, new)
    return True

def apply_lora_loralib(model: nn.Module,
                       target_module_names,
                       r: int, alpha: int, dropout: float,
                       keep_classifier_trainable: bool = True):
    """
    遍历模型，把名字在 target_module_names 的 nn.Linear 换成 lora.Linear。
    然后只训练 LoRA 参数（以及可选的分类头）。
    """
    replaced = 0

    def _recursive(module: nn.Module, prefix: str = ""):
        nonlocal replaced
        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            if name in target_module_names or any(
                full_name.endswith(tgt) or name.endswith(tgt) for tgt in target_module_names
            ):
                if isinstance(child, nn.Linear):
                    ok = _replace_linear_with_lora(module, name, r, alpha, dropout)
                    if ok:
                        replaced += 1
                        continue
            _recursive(child, full_name)

    _recursive(model)

    # 冻结全部参数
    for p in model.parameters():
        p.requires_grad = False

    # 仅打开 LoRA 参数
    lora.mark_only_lora_as_trainable(model)

    # 分类头仍然训练（DistilBERT：pre_classifier + classifier）
    if keep_classifier_trainable:
        for n, p in model.named_parameters():
            if "classifier" in n or "pre_classifier" in n:
                p.requires_grad = True

    print(f"[LoRA] replaced linear layers: {replaced}")
    print(f"[LoRA] now trainable params: {count_trainable_params(model):,}")
    return model, replaced

# =========================
# 主流程
# =========================
def main():
    parser = argparse.ArgumentParser(description="DistilBERT + LoRA (loralib) on AG News (legacy-friendly)")
    # baseline
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--dataset_name", type=str, default="ag_news")
    parser.add_argument("--output_dir", type=str, default="./results_lora")
    parser.add_argument("--save_dir", type=str, default="./saved_lora")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=128)

    # LoRA（loralib）
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_targets", type=str, default="q_lin,v_lin")  # DistilBERT 注意力默认
    parser.add_argument("--lora_merge_on_save", action="store_true",
                        help="保存前将 LoRA 合并到主权重（可选，默认不合并）")

    args = parser.parse_args()

    # LoRA / 全参
    mode = "LoRA" if args.use_lora else "Full FT"
    print(f"[Mode] {mode}")
    if args.use_lora:
        print(f"[LoRA cfg] r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}, targets={args.lora_targets}")

    set_seed(args.seed)
    device = get_device()

    print("Loading dataset...")
    dataset, num_labels, text_key, label_key = load_dataset_by_name(args.dataset_name)

    # 从训练集切出验证集（分层 stratify）
    split = dataset["train"].train_test_split(
        test_size=0.1, seed=args.seed, stratify_by_column=label_key
    )
    dataset["train"] = split["train"]
    dataset["validation"] = split["test"]

    print("Loading tokenizer & preprocessing...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    def preprocess(batch):
        return tokenizer(batch[text_key], truncation=True, max_length=args.max_length)

    # map 到三个 split（train/validation/test）
    tokenized = dataset.map(preprocess, batched=True)
    tokenized = tokenized.rename_column(label_key, "labels")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=num_labels
    )

    # LoRA 注入（仅当 --use_lora）
    lora_replaced = 0
    if args.use_lora:
        targets = [t.strip() for t in args.lora_targets.split(",") if t.strip()]
        model, lora_replaced = apply_lora_loralib(
            model,
            target_module_names=targets,
            r=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            keep_classifier_trainable=True,
        )

    model.to(device)
    print(f"Trainable params: {count_trainable_params(model):,}")

    # 指标
    try:
        import evaluate
        metric_acc = evaluate.load("accuracy")
        metric_f1  = evaluate.load("f1")
    except Exception as e:
        print("`evaluate` 未安装或加载失败，将使用简易 accuracy。原始错误：", e)
        metric_acc = None
        metric_f1  = None

    def compute_metrics(eval_pred):
        import numpy as np
        logits, labels = eval_pred
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        preds = np.argmax(logits, axis=-1)
        out = {}
        if metric_acc is not None:
            out["accuracy"] = metric_acc.compute(predictions=preds, references=labels)["accuracy"]
        else:
            out["accuracy"] = float((preds == labels).mean())

        if metric_f1 is not None:
            out["macro_f1"] = metric_f1.compute(predictions=preds, references=labels, average="macro")["f1"]

        return out

        # if metric_acc is not None:
        #     return metric_acc.compute(predictions=preds, references=labels)
        # else:
        #     return {"accuracy": float((preds == labels).mean())}

    # TrainingArguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        logging_dir=os.path.join(args.output_dir, "logs"),

        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,

        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=1,

        fp16=False,
        bf16=False,

        dataloader_pin_memory=False,
        dataloader_num_workers=0,

        logging_steps=100,
        seed=args.seed,
    )

    from transformers import TrainerCallback

    # 改动 4：Trainer 用 validation 做 eval；每个 epoch 末由回调触发一次评估
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],  
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EvalAtEpochEnd()],
    )

    # 记录训练时长/吞吐
    print("Start training...")
    t0 = time.time()
    train_output = trainer.train()
    t1 = time.time()
    train_runtime = t1 - t0

    num_train_samples = len(tokenized["train"])
    num_train_steps = math.ceil(num_train_samples / args.batch_size) * args.epochs
    steps_per_second = num_train_steps / train_runtime if train_runtime > 0 else 0.0
    samples_per_second = (num_train_samples * args.epochs) / train_runtime if train_runtime > 0 else 0.0

    print("Training finished.")
    print({
        "train_runtime_sec": train_runtime,
        "steps_per_second": steps_per_second,
        "samples_per_second": samples_per_second
    })

    # === 导出训练日志 + 画曲线===
    # 1) 导出 trainer.state.log_history 为 JSONL
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "training_log.jsonl"), "w") as f:
        for row in trainer.state.log_history:
            f.write(json.dumps(row) + "\n")
    print("Wrote:", os.path.join(args.output_dir, "training_log.jsonl"))

    # 2) 画 Train Loss + Val Acc + Val macro-F1（单 y 轴）
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MultipleLocator
        from collections import defaultdict, OrderedDict

        history = trainer.state.log_history

        # 1) 每个 epoch 的 Train Loss（取均值）
        loss_bins = defaultdict(list)
        for row in history:
            if "loss" in row and "epoch" in row:
                e = int(math.ceil(row["epoch"]))
                loss_bins[e].append(row["loss"])
        loss_by_epoch = OrderedDict(sorted((e, sum(v)/len(v)) for e, v in loss_bins.items()))
        epochs_loss = list(loss_by_epoch.keys())
        train_loss  = list(loss_by_epoch.values())

        # 2) 抓取 Val Acc / Val macro-F1
        val_acc_epochs, val_acc = [], []
        val_f1_epochs,  val_f1  = [], []
        for row in history:
            if "eval_accuracy" in row and "epoch" in row:
                val_acc_epochs.append(int(round(row["epoch"])))
                val_acc.append(row["eval_accuracy"])
            if "eval_macro_f1" in row and "epoch" in row:
                val_f1_epochs.append(int(round(row["epoch"])))
                val_f1.append(row["eval_macro_f1"])

        # 3) 单 y 轴作图
        plt.figure(figsize=(7,4))
        if epochs_loss:
            plt.plot(epochs_loss, train_loss, marker="^", linestyle="--", label="Train Loss")
        if val_acc_epochs:
            plt.plot(val_acc_epochs, val_acc, marker="o", label="Val Acc")
        if val_f1_epochs:
            plt.plot(val_f1_epochs, val_f1, marker="s", label="Val macro-F1")

        ax = plt.gca()
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Metric value")
        ax.set_ylim(0, 1.0)  
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.margins(x=0.05, y=0.05)
        plt.legend()
        plt.tight_layout()
        curve_path = os.path.join(args.output_dir, "training_curves.png")
        plt.savefig(curve_path, dpi=200); plt.close()
        print("Saved:", curve_path)

    except Exception as e:
        print("绘图失败：", e)


    # # === 画图结束 ===


    # 训练后分别评估 val & test，并保存文本指标
    print("Evaluating on validation set...")
    val_metrics = trainer.evaluate(eval_dataset=tokenized["validation"])
    print(val_metrics)

    print("Evaluating on test set...")
    test_metrics = trainer.evaluate(eval_dataset=tokenized["test"])
    print(test_metrics)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "val_metrics.txt"), "w") as f:
        for k, v in val_metrics.items():
            f.write(f"{k}: {v}\n")
    with open(os.path.join(args.output_dir, "test_metrics.txt"), "w") as f:
        for k, v in test_metrics.items():
            f.write(f"{k}: {v}\n")

    # 分类报告 & 混淆矩阵（基于测试集）
    try:
        # 依赖：pip install scikit-learn matplotlib
        from sklearn.metrics import classification_report, confusion_matrix
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # 1) 预测测试集
        pred = trainer.predict(tokenized["test"])
        y_true = pred.label_ids
        y_pred = pred.predictions.argmax(-1)

        # 2) 类别名（按 label id 顺序）
        class_names = dataset["train"].features["label"].names
        # 也可以用你固定的命名：
        # class_names = ["World", "Sports", "Business", "Sci/Tech"]

        # 3) 文本分类报告
        with open(os.path.join(args.output_dir, "classification_report.txt"), "w") as f:
            f.write(classification_report(y_true, y_pred, target_names=class_names, digits=4))

        # 4) 混淆矩阵 数值 + CSV
        cm = confusion_matrix(y_true, y_pred)                     # 计数版
        # cm = confusion_matrix(y_true, y_pred, normalize="true") # 想要百分比就用这一行
        np.savetxt(os.path.join(args.output_dir, "confusion_matrix.csv"),
                cm.astype(float), fmt="%.0f", delimiter=",")

        # 5) 画PNG图
        fig, ax = plt.subplots(figsize=(7.5, 7))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")

        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names, rotation=35, ha="right")
        ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")

        # 在格子里标数字（若用了 normalize="true" 改成 .2f 或百分比）
        fmt = "d"  # 或 ".2f"
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val = cm[i, j]
                ax.text(j, i, f"{val:{fmt}}",
                        ha="center", va="center",
                        color="white" if val > thresh else "black",
                        fontsize=11)

        plt.tight_layout()
        out_path = os.path.join(args.output_dir, "confusion_matrix.png")
        plt.savefig(out_path, dpi=200)
        plt.close()
        print("Saved:", out_path)

    except Exception as e:
        print("生成分类报告/混淆矩阵失败：", e)


    # 将关键信息写入 run_summary.json
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = count_trainable_params(model)
    run_info = {
        "mode": mode,
        "model_name": args.model_name,
        "dataset_name": args.dataset_name,
        "num_labels": num_labels,
        "max_length": args.max_length,
        "seed": args.seed,

        "use_lora": args.use_lora,
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "targets": [t for t in args.lora_targets.split(",") if t],
            "replaced_layers": lora_replaced,
        },

        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "trainable_ratio": float(trainable_params / total_params),

        "training_args": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        },

        "train_speed": {
            "train_runtime_sec": float(train_runtime),
            "steps_per_second": float(steps_per_second),
            "samples_per_second": float(samples_per_second),
        },

        "val_metrics": {k: float(v) for k, v in val_metrics.items()},
        "test_metrics": {k: float(v) for k, v in test_metrics.items()},
    }
    with open(os.path.join(args.output_dir, "run_summary.json"), "w") as f:
        json.dump(run_info, f, indent=2, ensure_ascii=False)

    # 保存模型（可选合并 LoRA）
    if args.use_lora and args.lora_merge_on_save:
        print("[LoRA] merging LoRA weights into base weights for saving...")
        lora.merge_lora_weights(model)
    else:
        print("[LoRA] (skip merge) saving with LoRA adapters still separate.")

    print(f"Saving model to: {args.save_dir}")
    trainer.save_model(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)

    print("Done.")

if __name__ == "__main__":
    main()
