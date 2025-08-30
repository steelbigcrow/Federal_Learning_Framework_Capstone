import os
import re
import math
import random
from collections import Counter
from typing import List, Tuple
import matplotlib.pyplot as plt
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
import loralib as lora


# ------------------------------
# Utils & Tokenization
# ------------------------------
def simple_tokenize(text: str) -> List[str]:
    # convert to lowercase + extract word characters
    return re.findall(r"[A-Za-z0-9']+", text.lower())

def build_vocab(texts: List[str], vocab_size: int = 30000, min_freq: int = 2):
    counter = Counter()
    for t in texts:
        counter.update(simple_tokenize(t))
    # special symbols
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
    #Text classification dataset class(fixed-length id sequence, label)
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
# Model: LoRA Transformer (from scratch)
# ------------------------------
class PositionalEncoding(nn.Module):
    #实现 Transformer 里的正弦位置编码,把“顺序感”注入到词向量
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (L, D) pe存放位置编码
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (L,1)把每个 token 的位置编号，变成一个列向量，方便后续和频率向量
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) #实现 不同维度用不同波长的 sin/cos 函数。
        pe[:, 0::2] = torch.sin(position * div_term) #偶数维度 → sin 波形
        pe[:, 1::2] = torch.cos(position * div_term) #奇数维度 → cos 波形
        self.register_buffer('pe', pe)  # 把算出来的pe存在模型里，不作为参数更新

    def forward(self, x):  # x: (B, L, D)要给每个 token 的 embedding 加上它对应的 位置向量
        L = x.size(1)
        return x + self.pe[:L].unsqueeze(0)  # (1,L,D)

class LoRA_MHA(nn.Module):#多头自注意力：让每个 token 去“看”序列里其它 token
    """定义一个 PyTorch 模块，继承 nn.Module。多头注意力（自实现），所有线性层用 loralib.Linear。把一个大维度向量拆成若干个小块，每个小块单独做注意力运算。"""
    def __init__(self, d_model: int, n_heads: int, r: int, alpha: int, lora_dropout: float, attn_dropout: float):
        super().__init__()

        assert d_model % n_heads == 0
        self.d_model = d_model #输入/输出的总维度（整层的隐藏通道数）
        self.n_heads = n_heads #注意力头数（把向量切几块）
        self.d_head = d_model // n_heads #每个头分到的维度大小

        self.q_proj = lora.Linear(d_model, d_model, r=r, lora_alpha=alpha, lora_dropout=lora_dropout, bias=True)
        self.k_proj = lora.Linear(d_model, d_model, r=r, lora_alpha=alpha, lora_dropout=lora_dropout, bias=True)
        self.v_proj = lora.Linear(d_model, d_model, r=r, lora_alpha=alpha, lora_dropout=lora_dropout, bias=True)
        self.o_proj = lora.Linear(d_model, d_model, r=r, lora_alpha=alpha, lora_dropout=lora_dropout, bias=True)

        self.attn_drop = nn.Dropout(attn_dropout)#防止注意力分布过拟合

    def forward(self, x, attn_mask):  # x: (B,L,D); attn_mask: (B,1,1,L)  1=keep, 0=mask
        #Q/K/V 线性投影 + 把隐藏维 D 切分成 n_heads 个小头
        B, L, D = x.size()
        q = self.q_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)  # (B,H,L,dh)
        k = self.k_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)  # (B,H,L,dh)
        v = self.v_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)  # (B,H,L,dh)

        # 注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)  # (B,H,L,L)
        if attn_mask is not None:
            # 将被 mask 的位置设为非常小
            scores = scores.masked_fill(attn_mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)

        ctx = torch.matmul(attn, v)  # (B,H,L,dh)
        ctx = ctx.transpose(1, 2).contiguous().view(B, L, D)  # (B,L,D)
        out = self.o_proj(ctx)  # (B,L,D)
        return out

class LoRA_FFN(nn.Module):#前馈网络：对每个 token 自身做非线性变换（位置内 MLP）
    def __init__(self, d_model: int, d_ff: int, r: int, alpha: int, lora_dropout: float, dropout: float):
        super().__init__()
        #两个全连接层换成lora；fc1：把输入从 d_model 投影到 d_ff fc2：把中间层再投影回 d_model
        self.fc1 = lora.Linear(d_model, d_ff, r=r, lora_alpha=alpha, lora_dropout=lora_dropout, bias=True)
        self.fc2 = lora.Linear(d_ff, d_model, r=r, lora_alpha=alpha, lora_dropout=lora_dropout, bias=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.fc2(F.gelu(self.fc1(x))))

class LoRA_TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, r, alpha, lora_dropout, attn_dropout, resid_dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.mha = LoRA_MHA(d_model, n_heads, r, alpha, lora_dropout, attn_dropout)
        self.drop1 = nn.Dropout(resid_dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = LoRA_FFN(d_model, d_ff, r, alpha, lora_dropout, resid_dropout)
        self.drop2 = nn.Dropout(resid_dropout)

    def forward(self, x, attn_mask):
        # Self-Attention自注意力子层
        h = self.mha(self.ln1(x), attn_mask)#归一化后的输入 → 多头注意力 → 上下文表示
        x = x + self.drop1(h)#把注意力结果加回原输入（残差）+ dropout 正则化
        # FFN前馈子层
        h2 = self.ffn(self.ln2(x))
        x = x + self.drop2(h2)
        return x

class LoRA_TransformerClassifier(nn.Module):
    #顶层封装：Embedding + PositionalEncoding → n 层 LoRA Transformer → LayerNorm → 分类头
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=2, d_ff=256,
                 max_len=256, num_classes=2, r=8, alpha=16, lora_dropout=0.1,
                 attn_dropout=0.1, resid_dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = PositionalEncoding(d_model, max_len=max_len)
        self.layers = nn.ModuleList([#LoRA 多头注意力 (LoRA_MHA);LoRA 前馈网络 (LoRA_FFN);残差 + LayerNorm
            LoRA_TransformerEncoderLayer(
                d_model, n_heads, d_ff, r, alpha, lora_dropout, attn_dropout, resid_dropout
            )
            for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)  # 分类头不加 LoRA（也可以换成 lora.Linear 看需求）

    def forward(self, input_ids):  # (B,L)
        mask = (input_ids != 0).unsqueeze(1).unsqueeze(2)  # (B,1,1,L) 1=keep
        x = self.embed(input_ids)  # (B,L,D)
        x = self.pos(x)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.ln(x)  # (B,L,D)
        # mean pooling (忽略 padding)
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
    step_losses = []  # 新增：记录每个 step 的 loss
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
        step_losses.append(loss.item())  # 记录本 step 的 loss

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, step_losses


# ------------------------------
# Main
# ------------------------------
def main():
    # 超参（可用环境变量覆盖）
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

    # LoRA 超参
    LORA_R = int(os.environ.get("LORA_R", "8"))
    LORA_ALPHA = int(os.environ.get("LORA_ALPHA", "16"))
    LORA_DROPOUT = float(os.environ.get("LORA_DROPOUT", "0.1"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> Device: {device}")
    if device.type == "cuda":
        print(">>> GPU:", torch.cuda.get_device_name(0))

    # 1) 加载 IMDB
    print("Loading IMDB ...")
    ds = load_dataset("imdb")
    train_texts = ds["train"]["text"]
    train_labels = ds["train"]["label"]
    test_texts = ds["test"]["text"]
    test_labels = ds["test"]["label"]

    # 2) 构建词表 & 数值化
    print("Building vocab ...")
    stoi, itos = build_vocab(train_texts, vocab_size=vocab_size, min_freq=2)

    train_pairs = list(zip(train_texts, train_labels))
    test_pairs = list(zip(test_texts, test_labels))
    train_loader = make_dataloader(train_pairs, stoi, max_len, batch_size, shuffle=True)
    test_loader  = make_dataloader(test_pairs,  stoi, max_len, batch_size, shuffle=False)

    # 3) 构建 LoRA Transformer
    model = LoRA_TransformerClassifier(
        vocab_size=len(stoi),
        d_model=d_model, n_heads=n_heads, n_layers=n_layers, d_ff=d_ff,
        max_len=max_len, num_classes=2,
        r=LORA_R, alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        attn_dropout=0.1, resid_dropout=0.1
    ).to(device)

    # 4) 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # 5) 训练
    os.makedirs("./outputs/loralib_scratch", exist_ok=True)

    history = {"epoch": [], "train_loss": [], "val_acc": [], "val_f1": []}
    all_step_losses = []   # 跨 epoch 的 step loss
    global_steps = []      # 跨 epoch 的全局 step 编号（从 1 开始累加）
    best_acc = 0.0
    step_counter = 0

    for epoch in range(1, epochs + 1):
        train_loss, step_losses = train_one_epoch(model, train_loader, optimizer, device)
        acc, f1 = evaluate(model, test_loader, device)
        print(f"[Epoch {epoch}/{epochs}] loss={train_loss:.4f} | acc={acc:.4f} | f1={f1:.4f}")

        # 记录 epoch 级指标
        history["epoch"].append(epoch)
        history["train_loss"].append(float(train_loss))
        history["val_acc"].append(float(acc))
        history["val_f1"].append(float(f1))

        # 记录 step 级 loss（拼到全局曲线）
        for sl in step_losses:
            step_counter += 1
            all_step_losses.append(float(sl))
            global_steps.append(step_counter)

        # 保存最好模型
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "./outputs/loralib_scratch/model.pt")
            print("↳ New best! Saved to ./outputs/loralib_scratch/model.pt")

    print(f"Best Acc: {best_acc:.4f}")

    # ---------- 可视化 ----------
    import matplotlib.pyplot as plt
    import json

    # 保存历史到 json
    with open("./outputs/loralib_scratch/history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    # 1) 训练 Loss（按 epoch）
    plt.figure()
    plt.title("Training Loss (per epoch)")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.plot(history["epoch"], history["train_loss"], marker="o")
    plt.grid(True); plt.tight_layout()
    plt.savefig("./outputs/loralib_scratch/loss_curve.png", dpi=150)
    plt.close()

    # 2) 训练 Loss（按 step，跨 epoch 连续）
    if len(global_steps) > 0:
        plt.figure()
        plt.title("Training Loss (per step)")
        plt.xlabel("Global Step"); plt.ylabel("Loss")
        plt.plot(global_steps, all_step_losses, linewidth=1)
        plt.grid(True); plt.tight_layout()
        plt.savefig("./outputs/loralib_scratch/loss_per_step.png", dpi=150)
        plt.close()

    # 3) 验证 ACC & F1（按 epoch）
    plt.figure()
    plt.title("Validation Metrics")
    plt.xlabel("Epoch"); plt.ylabel("Score")
    plt.plot(history["epoch"], history["val_acc"], marker="o", label="ACC")
    plt.plot(history["epoch"], history["val_f1"], marker="o", label="F1")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig("./outputs/loralib_scratch/val_metrics.png", dpi=150)
    plt.close()

    print("Saved curves to ./outputs/loralib_scratch/:",
          "loss_curve.png, loss_per_step.png, val_metrics.png")



if __name__ == "__main__":
    main()