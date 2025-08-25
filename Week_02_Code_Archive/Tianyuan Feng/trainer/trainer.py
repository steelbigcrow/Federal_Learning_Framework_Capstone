import os
import torch
import torch.nn as nn
from tqdm import tqdm
import loralib as lora
import copy
from torch.optim.lr_scheduler import ReduceLROnPlateau

class Trainer:
    def __init__(self, cfg, device):
        # 包含了所有超参数和配置的对象。
        self.cfg = cfg
        # 指定在GPU上进行训练。
        self.device = device

    @staticmethod
    def count_params(model):
        # 计算模型的总参数量
        total = sum(p.numel() for p in model.parameters())
        # 只计算那些需要梯度更新的参数（可训练参数）
        # 这对于LoRA微调尤其重要，可以清晰地看到可训练参数的比例。
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total, trainable, (trainable / total if total > 0 else 0.0)

    def fit(self, model, train_loader, val_loader):
        # 将模型带到GPU
        model = model.to(self.device)
        # 确定评分标准：BCEWithLogitsLoss 是一个适用于二分类任务的、数值上很稳定的损失函数。
        crit = nn.BCEWithLogitsLoss()
        # AdamW 是一种先进的优化器，负责根据训练反馈来调整模型的肌肉（权重）。
        opt = torch.optim.AdamW(model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        # ReduceLROnPlateau 会监控val_loss，如果model表现停滞不前，就调整降低学习率。
        scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=1)

        total, trainable, ratio = self.count_params(model)
        print(f"Parameters  total={total:,}  trainable={trainable:,}  ratio={ratio:.3f}")
        is_binary = (model.head[-1].out_features == 1)
        best_val_loss = float('inf')
        best_model_state = None


        # 2. 按Epoch进行训练循环
        for epoch in range(1, self.cfg.epochs + 1):
            # 命令模型进入“训练模式”，这会启用Dropout等正则化手段。
            model.train()
            running_loss = 0.0
            # 为训练过程创建一个进度条
            pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Training]")
            # 遍历每一个batch
            for x, lengths, y in pbar:
                x, lengths, y = x.to(self.device), lengths.to(self.device), y.to(self.device)
                # 清空上一轮的梯度
                opt.zero_grad(set_to_none=True)
                # 前向传播
                logits = model(x, lengths)
                # 计算损失
                loss = crit(logits.squeeze(-1), y.float()) if is_binary else crit(logits, y)
                # 反向传播，计算梯度
                loss.backward()
                if self.cfg.grad_clip is not None:
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg.grad_clip)
                # 优化器更新权重
                opt.step()
                running_loss += loss.item() * x.size(0)
                # 在进度条上更新平均得分
                pbar.set_postfix(loss=f"{running_loss/len(train_loader.dataset):.4f}")
            train_loss = running_loss / len(train_loader.dataset)
            print(f"[Epoch {epoch}] train_loss={train_loss:.4f}")
            # 关闭Dropout，保证结果稳定。
            if val_loader:
                model.eval()
                val_loss = 0.0
                # 不计算梯度
                with torch.no_grad():
                    # 前向传播和计算损失
                    for x, lengths, y in tqdm(val_loader, desc=f"Epoch {epoch} [Validation]"):
                        x, lengths, y = x.to(self.device), lengths.to(self.device), y.to(self.device)
                        logits = model(x, lengths)
                        loss = crit(logits.squeeze(-1), y.float()) if is_binary else crit(logits, y)
                        # 调整学习率
                        val_loss += loss.item() * x.size(0)
                val_loss /= len(val_loader.dataset)
                print(f"[Epoch {epoch}] val_loss={val_loss:.4f}")
                scheduler.step(val_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    if self.cfg.save_best:
                        best_model_state = copy.deepcopy(model.state_dict())
                        print(f"  -> New best model saved with val_loss: {best_val_loss:.4f}")
        if self.cfg.save_best and best_model_state:
            model.load_state_dict(best_model_state)
            print("Loaded best model state for final testing and export.")
        return model

    def test(self, model, loader):
        model.eval()
        correct = 0
        with torch.no_grad():
            for x, lengths, y in tqdm(loader, desc="Final Testing"):
                x, lengths, y = x.to(self.device), lengths.to(self.device), y.to(self.device)
                logits = model(x, lengths)
                # 将裁判的原始分（logits）转化为0或1的最终判决
                preds = torch.round(torch.sigmoid(logits.squeeze(-1)))
                # 统计判断正确的数量
                correct += (preds == y).sum().item()
        # 计算最终正确率
        accuracy = correct / len(loader.dataset)
        print(f"Test Accuracy: {accuracy:.4f}")

    def export(self, model, tag):
        # 处理LoRA相关的特殊保存逻辑
        if tag == "lstm_lora" and self.cfg.merge_lora:
            print("Merging LoRA weights into base (loralib.merge_lora_weights)...")
            lora.merge_lora_weights(model)
        # 确保路径存在
        os.makedirs(self.cfg.artifacts_dir, exist_ok=True)
        # 确定.pt文件的位置
        path = os.path.join(self.cfg.artifacts_dir, f"{tag}_imdb.pt")
        # 将模型权重存入指定位置
        torch.save(model.state_dict(), path)
        print(f"Saved model to {path}")
        print(f"Vocab saved to {os.path.join(self.cfg.artifacts_dir, 'vocab.json')}")

        if tag == "lstm_lora" and self.cfg.save_lora_only:
            lora_path = os.path.join(self.cfg.artifacts_dir, f"{tag}_lora_only.pt")
            torch.save(lora.lora_state_dict(model), lora_path)
            print(f"Saved LoRA-only params to {lora_path}")