# transformer_imdb_plain.py
import os
import re
import math
import random
from collections import Counter
from typing import List, Tuple

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score

# ------------------------------
# Utils & Tokenization
# ------------------------------
def simple_tokenize(text: str) -> List[str]:
    # lowercase + extract word characters
    return re.findall(r"[A-Za-z0-9']+", text.lower())

def build_vocab(texts: List[str], vocab_size: int = 30000, min_freq: int = 2):
    counter = Counter()
    for t in texts:
        counter.update(simple_tokenize(t))
    # special tokens
    stoi = {"<pad>": 0, "<unk>": 1}
    for tok, freq in counter.most_common():
        if tok in stoi:
            continue
        if freq < min_freq:
            continue
        stoi[tok] = len(stoi)
        if len(stoi) >= vocab_size:
            break
    itos = {i: s for s, i in stoi.items()}
    return stoi, itos

def encode(text: str, stoi: dict) -> List[int]:
    return [stoi.get(tok, 1) for tok in simple_tokenize(text)]  # 1 = <unk>

def pad_trunc(ids: List[int], max_len: int, pad_id: int = 0) -> List[int]:
    if len(ids) >= max_len:
        return ids[:max_len]
    return ids + [pad_id] * (max_len - len(ids))

# ------------------------------
# Data Loader
# ------------------------------
class TextClsDataset(torch.utils.data.Dataset):
    """Text classification dataset (fixed-length id sequence, label)"""
    def __init__(self, pairs: List[Tuple[str, int]], stoi: dict, max_len: int):
        self.pairs = pairs
        self.stoi = stoi
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        text, label = self.pairs[idx]
        ids = encode(text, self.stoi)
        ids = pad_trunc(ids, self.max_len, pad_id=0)
        return torch.tensor(ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)

def make_dataloader(pairs, stoi, max_len, batch_size, shuffle):
    ds = TextClsDataset(pairs, stoi, max_len)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)

# ------------------------------
# Model: Plain Transformer (no LoRA)
# ------------------------------
class PositionalEncoding(nn.Module):
    """正弦位置编码：把顺序注入到词向量"""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)   # (L,1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # even dims
        pe[:, 1::2] = torch.cos(position * div_term)  # odd dims
        self.register_buffer('pe', pe)  # 不更新的缓冲参数

    def forward(self, x):  # x: (B,L,D)
        L = x.size(1)
        return x + self.pe[:L].unsqueeze(0)  # (1,L,D)

class MHA(nn.Module):
    """多头自注意力（Q/K/V/O 全用 nn.Linear）"""
    def __init__(self, d_model: int, n_heads: int, attn_dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.o_proj = nn.Linear(d_model, d_model, bias=True)

        self.attn_drop = nn.Dropout(attn_dropout)

    def forward(self, x, attn_mask):  # x: (B,L,D); attn_mask: (B,1,1,L)
        B, L, D = x.size()
        q = self.q_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)  # (B,H,L,dh)
        k = self.k_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)  # (B,H,L,dh)
        v = self.v_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)  # (B,H,L,dh)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)     # (B,H,L,L)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)

        ctx = torch.matmul(attn, v)                                                # (B,H,L,dh)
        ctx = ctx.transpose(1, 2).contiguous().view(B, L, D)                        # (B,L,D)
        out = self.o_proj(ctx)                                                      # (B,L,D)
        return out

class FFN(nn.Module):
    """前馈网络：两层 nn.Linear + GELU"""
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff, bias=True)
        self.fc2 = nn.Linear(d_ff, d_model, bias=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.fc2(F.gelu(self.fc1(x))))

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, attn_dropout, resid_dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.mha = MHA(d_model, n_heads, attn_dropout)
        self.drop1 = nn.Dropout(resid_dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model, d_ff, resid_dropout)
        self.drop2 = nn.Dropout(resid_dropout)

    def forward(self, x, attn_mask):
        # Self-Attention
        h = self.mha(self.ln1(x), attn_mask)
        x = x + self.drop1(h)
        # FFN
        h2 = self.ffn(self.ln2(x))
        x = x + self.drop2(h2)
        return x

class TransformerClassifier(nn.Module):
    """Embedding + PosEnc → N×Encoder → LN → mean-pool → Linear head"""
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=2, d_ff=256,
                 max_len=256, num_classes=2, attn_dropout=0.1, resid_dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = PositionalEncoding(d_model, max_len=max_len)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, attn_dropout, resid_dropout)
            for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, input_ids):  # (B,L)
        mask = (input_ids != 0).unsqueeze(1).unsqueeze(2)  # (B,1,1,L) 1=keep
        x = self.embed(input_ids)  # (B,L,D)
        x = self.pos(x)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.ln(x)  # (B,L,D)

        # mean pooling (ignore pad)
        mask_float = (input_ids != 0).float().unsqueeze(-1)  # (B,L,1)
        x_sum = (x * mask_float).sum(dim=1)                  # (B,D)
        lengths = mask_float.sum(dim=1).clamp(min=1e-6)      # (B,1)
        pooled = x_sum / lengths                             # (B,D)
        logits = self.head(pooled)                           # (B,C)
        return logits

# ------------------------------
# Train / Eval
# ------------------------------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for ids, labels in loader:
        ids = ids.to(device)
        labels = labels.to(device)
        logits = model(ids)
        preds = logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    return acc, f1

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    step_losses = []  # record per-step loss
    for ids, labels in loader:
        ids = ids.to(device)
        labels = labels.to(device)
        logits = model(ids)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * ids.size(0)
        step_losses.append(loss.item())
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, step_losses

def count_trainable_params(model):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total

# ------------------------------
# Main
# ------------------------------
def main():
    # Hyperparams (can be overridden by env)
    seed = int(os.environ.get("SEED", "42"))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    max_len = int(os.environ.get("MAX_LENGTH", "256"))
    vocab_size = int(os.environ.get("VOCAB_SIZE", "30000"))
    batch_size = int(os.environ.get("BATCH_SIZE", "32"))
    epochs = int(os.environ.get("EPOCHS", "12"))
    lr = float(os.environ.get("LR", "3e-4"))

    d_model = int(os.environ.get("D_MODEL", "128"))
    n_heads = int(os.environ.get("N_HEADS", "4"))
    n_layers = int(os.environ.get("N_LAYERS", "2"))
    d_ff = int(os.environ.get("D_FF", "256"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> Device: {device}")
    if device.type == "cuda":
        print(">>> GPU:", torch.cuda.get_device_name(0))

    # 1) Load dataset
    print("Loading IMDB ...")
    ds = load_dataset("imdb")
    train_texts = ds["train"]["text"]
    train_labels = ds["train"]["label"]
    test_texts = ds["test"]["text"]
    test_labels = ds["test"]["label"]

    # 2) Build vocab & numericalize
    print("Building vocab ...")
    stoi, itos = build_vocab(train_texts, vocab_size=vocab_size, min_freq=2)

    train_pairs = list(zip(train_texts, train_labels))
    test_pairs = list(zip(test_texts, test_labels))
    train_loader = make_dataloader(train_pairs, stoi, max_len, batch_size, shuffle=True)
    test_loader  = make_dataloader(test_pairs,  stoi, max_len, batch_size, shuffle=False)

    # 3) Build Plain Transformer (no LoRA)
    model = TransformerClassifier(
        vocab_size=len(stoi),
        d_model=d_model, n_heads=n_heads, n_layers=n_layers, d_ff=d_ff,
        max_len=max_len, num_classes=2,
        attn_dropout=0.1, resid_dropout=0.1
    ).to(device)

    # 额外打印：可训练参数量
    print("Trainable params:", count_trainable_params(model))

    # 4) Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # 5) Train
    os.makedirs("./outputs/plain_transformer", exist_ok=True)
    history = {"epoch": [], "train_loss": [], "val_acc": [], "val_f1": []}
    all_step_losses, global_steps = [], []
    best_acc = 0.0; step_counter = 0

    for epoch in range(1, epochs + 1):
        train_loss, step_losses = train_one_epoch(model, train_loader, optimizer, device)
        acc, f1 = evaluate(model, test_loader, device)
        print(f"[Epoch {epoch}/{epochs}] loss={train_loss:.4f} | acc={acc:.4f} | f1={f1:.4f}")

        history["epoch"].append(epoch)
        history["train_loss"].append(float(train_loss))
        history["val_acc"].append(float(acc))
        history["val_f1"].append(float(f1))

        for sl in step_losses:
            step_counter += 1
            all_step_losses.append(float(sl))
            global_steps.append(step_counter)

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "./outputs/plain_transformer/model.pt")
            print("↳ New best! Saved to ./outputs/plain_transformer/model.pt")

    print(f"Best Acc: {best_acc:.4f}")

    # ----- Visualization -----
    with open("./outputs/plain_transformer/history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    # 1) epoch loss
    plt.figure()
    plt.title("Training Loss (per epoch)")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.plot(history["epoch"], history["train_loss"], marker="o")
    plt.grid(True); plt.tight_layout()
    plt.savefig("./outputs/plain_transformer/loss_curve.png", dpi=150)
    plt.close()

    # 2) per-step loss
    if len(global_steps) > 0:
        plt.figure()
        plt.title("Training Loss (per step)")
        plt.xlabel("Global Step"); plt.ylabel("Loss")
        plt.plot(global_steps, all_step_losses, linewidth=1)
        plt.grid(True); plt.tight_layout()
        plt.savefig("./outputs/plain_transformer/loss_per_step.png", dpi=150)
        plt.close()

    # 3) val metrics
    plt.figure()
    plt.title("Validation Metrics")
    plt.xlabel("Epoch"); plt.ylabel("Score")
    plt.plot(history["epoch"], history["val_acc"], marker="o", label="ACC")
    plt.plot(history["epoch"], history["val_f1"], marker="o", label="F1")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig("./outputs/plain_transformer/val_metrics.png", dpi=150)
    plt.close()

    print("Saved curves to ./outputs/plain_transformer/:",
          "loss_curve.png, loss_per_step.png, val_metrics.png")

if __name__ == "__main__":
    main()
