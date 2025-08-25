# engine.py
from typing import Optional, Dict, Any, Tuple, Union
import time

import torch
import torch.nn as nn


class Trainer:
    """
    通用训练引擎，兼容预训练与 LoRA 微调。
    - 支持可选 AMP（混合精度）
    - 支持可选梯度裁剪
    - 支持调度器按 batch 或按 epoch 调用 step
    """
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        scheduler: Optional[Any] = None,
        scheduler_step_per_batch: bool = True,
        grad_clip: Optional[float] = None,
        use_amp: bool = False,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.scheduler_step_per_batch = scheduler_step_per_batch
        self.grad_clip = grad_clip
        # 仅在 CUDA 上启用 AMP；CPU 上即便传入 use_amp 也禁用，防止 autocast("cuda") 报错
        self.amp_device_type = "cuda" if str(device.type) == "cuda" else "cpu"
        self.use_amp = bool(use_amp and self.amp_device_type == "cuda")

        # torch.cuda.amp.GradScaler 已弃用，使用 torch.amp.GradScaler(device_type, ...)
        self.scaler = torch.amp.GradScaler(self.amp_device_type, enabled=self.use_amp)

    @staticmethod
    def _unpack_batch(batch: Union[Tuple[torch.Tensor, ...], list]):
        # 兼容 (x, y) 或 (x, y, lengths) 的 batch 结构
        if len(batch) == 2:
            texts, labels = batch
            lengths = None
        elif len(batch) == 3:
            texts, labels, lengths = batch
        else:
            raise ValueError("Batch must be (texts, labels) or (texts, labels, lengths).")
        return texts, labels, lengths

    @staticmethod
    def _batch_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
        preds = logits.argmax(dim=1)
        return (preds == labels).float().mean().item()

    def fit(
        self,
        train_loader,
        val_loader=None,
        num_epochs: int = 1,
        log_interval: Optional[int] = None,
    ):
        history = []
        for epoch in range(1, num_epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            epoch_acc = 0.0
            epoch_count = 0
            t0 = time.time()

            for step, batch in enumerate(train_loader, 1):
                texts, labels, lengths = self._unpack_batch(batch)
                texts = texts.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                # lengths 可选；模型内部会自动按 PAD 计算长度，因此这里不一定要传
                # 如果你希望传入：logits = self.model(texts, lengths=lengths)

                self.optimizer.zero_grad(set_to_none=True)

                if self.use_amp:
                    with torch.amp.autocast(self.amp_device_type, enabled=self.use_amp):
                        logits = self.model(texts)
                        loss = self.criterion(logits, labels)
                    self.scaler.scale(loss).backward()
                    if self.grad_clip is not None:
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    logits = self.model(texts)
                    loss = self.criterion(logits, labels)
                    loss.backward()
                    if self.grad_clip is not None:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optimizer.step()

                if self.scheduler is not None and self.scheduler_step_per_batch:
                    self.scheduler.step()

                with torch.no_grad():
                    acc = self._batch_accuracy(logits, labels)

                batch_size = labels.size(0)
                epoch_loss += loss.item() * batch_size
                epoch_acc += acc * batch_size
                epoch_count += batch_size

                if log_interval is not None and step % log_interval == 0:
                    print(f"Epoch {epoch}/{num_epochs} | Step {step}/{len(train_loader)} "
                          f"| Loss {loss.item():.4f} | Acc {acc * 100:.2f}%")

            train_loss = epoch_loss / max(1, epoch_count)
            train_acc = epoch_acc / max(1, epoch_count)
            elapsed = time.time() - t0

            # 按 epoch 调度器
            if self.scheduler is not None and not self.scheduler_step_per_batch:
                self.scheduler.step()

            print(f"[Train] Epoch {epoch}: Avg Loss={train_loss:.4f} | Avg Acc={train_acc * 100:.2f}% "
                  f"| Time {elapsed:.1f}s")

            metrics = {"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc}

            if val_loader is not None:
                eval_metrics = self.evaluate(val_loader)
                print(f"[Eval ] Epoch {epoch}: "
                      f"Loss={eval_metrics['loss']:.4f} | Acc={eval_metrics['acc'] * 100:.2f}%")
                metrics.update({f"val_{k}": v for k, v in eval_metrics.items()})

            history.append(metrics)

        return history

    def evaluate(self, data_loader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        total_acc = 0.0
        total_count = 0

        with torch.no_grad():
            for batch in data_loader:
                texts, labels, lengths = self._unpack_batch(batch)
                texts = texts.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                logits = self.model(texts)
                loss = self.criterion(logits, labels)
                acc = self._batch_accuracy(logits, labels)

                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size
                total_acc += acc * batch_size
                total_count += batch_size

        avg_loss = total_loss / max(1, total_count)
        avg_acc = total_acc / max(1, total_count)
        return {"loss": avg_loss, "acc": avg_acc}

    def save_checkpoint(self, path: str, extra: Optional[Dict[str, Any]] = None) -> None:
        ckpt = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "criterion": getattr(self.criterion, "state_dict", lambda: {})(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "scaler": self.scaler.state_dict() if self.use_amp else None,
            "extra": extra or {},
        }
        torch.save(ckpt, path)

    def load_checkpoint(self, path: str, map_location: str = "cpu", load_optimizer: bool = True) -> None:
        ckpt = torch.load(path, map_location=map_location)
        self.model.load_state_dict(ckpt["model"])
        if load_optimizer and "optimizer" in ckpt and ckpt["optimizer"] is not None:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt and ckpt["scheduler"] is not None and self.scheduler is not None:
            self.scheduler.load_state_dict(ckpt["scheduler"])
        if self.use_amp and "scaler" in ckpt and ckpt["scaler"] is not None:
            self.scaler.load_state_dict(ckpt["scaler"])