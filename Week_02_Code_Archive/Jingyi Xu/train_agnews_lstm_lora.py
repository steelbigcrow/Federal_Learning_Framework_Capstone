import re
import argparse
import random
from collections import Counter
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset


# ------------------- Utils: device & seed -------------------
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_device()

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ------------------- Data: tokenize / vocab -------------------
def tokenize(text: str) -> List[str]:
    return re.findall(r"\w+|[^\w\s]", text.lower())

def yield_tokens(dataset_split):
    for ex in dataset_split:
        yield tokenize(ex["text"])

SPECIALS = ["<pad>", "<unk>"]
PAD, UNK = 0, 1

def build_vocab(dataset_split, min_freq=2):
    counter = Counter()
    for tokens in yield_tokens(dataset_split):
        counter.update(tokens)
    itos = SPECIALS[:]
    for tok, freq in counter.items():
        if freq >= min_freq:
            itos.append(tok)
    stoi = {tok: i for i, tok in enumerate(itos)}
    return itos, stoi

def encode(text: str, stoi) -> List[int]:
    return [stoi.get(t, UNK) for t in tokenize(text)]

class AGNewsTorch(Dataset):
    def __init__(self, hf_split, stoi):
        self.data = hf_split
        self.stoi = stoi

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        ids = encode(ex["text"], self.stoi)
        y = int(ex["label"])  # already 0..3
        return torch.tensor(ids, dtype=torch.long), torch.tensor(y, dtype=torch.long)

def collate_batch(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
    ids_list, label_list, lengths = [], [], []
    for ids, y in batch:
        ids_list.append(ids)
        label_list.append(y)
        lengths.append(len(ids))
    max_len = max(lengths)
    padded = torch.full((len(ids_list), max_len), PAD, dtype=torch.long)
    for i, ids in enumerate(ids_list):
        padded[i, :len(ids)] = ids
    lengths = torch.tensor(lengths, dtype=torch.long)
    labels = torch.stack(label_list)
    return padded, lengths, labels


# ------------------- LoRA modules -------------------
class LoRAAdapter(nn.Module):
    """
    残差式 LoRA：对最后一维做线性低秩映射，然后加回输入（x + Δx）。
    适合放在 embedding 之后 (B, T, E) → (B, T, E)。
    """
    def __init__(self, dim: int, rank: int = 8, alpha: int = 16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        if rank > 0:
            self.A = nn.Parameter(torch.zeros(dim, rank))
            self.B = nn.Parameter(torch.zeros(rank, dim))
            # LoRA 常用：A 随机小值，B 全零，保证初始等价于恒等映射
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5)) if rank > 0 else None
            with torch.no_grad():
                self.B.zero_()
            self.scaling = alpha / rank
        else:
            self.register_parameter("A", None)
            self.register_parameter("B", None)
            self.scaling = 0.0

    def forward(self, x):
        if self.rank == 0:
            return x
        # x: (..., dim)
        delta = x @ self.A @ self.B
        return x + self.scaling * delta


class LoRALinearOnTop(nn.Module):
    """
    给已有的 Linear 加一个 LoRA 的 ΔW： y = base(x) + (x @ A @ B) * scaling
    冻结 base 的权重，仅训练 A/B。
    """
    def __init__(self, base_linear: nn.Linear, rank: int = 8, alpha: int = 16):
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False

        in_f = base_linear.in_features
        out_f = base_linear.out_features
        self.rank = rank
        if rank > 0:
            self.A = nn.Parameter(torch.zeros(in_f, rank))
            self.B = nn.Parameter(torch.zeros(rank, out_f))
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
            with torch.no_grad():
                self.B.zero_()
            self.scaling = alpha / rank
        else:
            self.register_parameter("A", None)
            self.register_parameter("B", None)
            self.scaling = 0.0

    def forward(self, x):
        out = self.base(x)
        if self.rank == 0:
            return out
        delta = x @ self.A @ self.B
        return out + self.scaling * delta


# ------------------- Model: baseline LSTM -------------------
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, num_classes, pad_idx,
                 bidirectional=True, dropout=0.4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), num_classes)
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, input_ids, lengths):
        emb = self.embedding(input_ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        if self.bidirectional:
            h_fw = h_n[-2, :, :]
            h_bw = h_n[-1, :, :]
            feat = self.dropout(torch.cat([h_fw, h_bw], dim=1))
        else:
            feat = self.dropout(h_n[-1, :, :])
        return self.fc(feat)


# ------------------- LoRA-wrapped classifier -------------------
class LSTMClassifierWithLoRA(LSTMClassifier):
    """
    在 embedding 输出后加 LoRA 残差适配器（维持形状不变）；
    在分类头 Linear 上叠加 LoRA ΔW。
    冻结 embedding 和 lstm 的参数，仅训练 LoRA + (可选) 分类头 LoRA。
    """
    def __init__(self, *args, lora_rank=8, lora_alpha=16, **kwargs):
        super().__init__(*args, **kwargs)
        # 1) 输入侧 LoRA（残差）：shape 保持 [B,T,E]
        embed_dim = self.embedding.embedding_dim
        self.lora_in = LoRAAdapter(embed_dim, rank=lora_rank, alpha=lora_alpha)

        # 2) 分类头 LoRA：y = base(x) + ΔW x
        self.fc = LoRALinearOnTop(self.fc, rank=lora_rank, alpha=lora_alpha)

        # 3) 冻结大块权重：embedding + lstm
        for p in self.embedding.parameters():
            p.requires_grad = False
        for p in self.lstm.parameters():
            p.requires_grad = False

    def forward(self, input_ids, lengths):
        emb = self.embedding(input_ids)           # [B,T,E]
        emb = self.lora_in(emb)                   # LoRA 适配后的表示
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        if self.bidirectional:
            h_fw = h_n[-2, :, :]
            h_bw = h_n[-1, :, :]
            feat = self.dropout(torch.cat([h_fw, h_bw], dim=1))
        else:
            feat = self.dropout(h_n[-1, :, :])
        return self.fc(feat)


# ------------------- Train / Eval -------------------
def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable, trainable / total

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    tot_loss, tot_correct, tot_ex = 0.0, 0, 0
    for input_ids, lengths, labels in loader:
        input_ids, lengths, labels = input_ids.to(device), lengths.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(input_ids, lengths)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        tot_loss += loss.item() * input_ids.size(0)
        tot_correct += (logits.argmax(1) == labels).sum().item()
        tot_ex += input_ids.size(0)
    return tot_loss / tot_ex, tot_correct / tot_ex

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    tot_loss, tot_correct, tot_ex = 0.0, 0, 0
    for input_ids, lengths, labels in loader:
        input_ids, lengths, labels = input_ids.to(device), lengths.to(device), labels.to(device)
        logits = model(input_ids, lengths)
        loss = criterion(logits, labels)
        tot_loss += loss.item() * input_ids.size(0)
        tot_correct += (logits.argmax(1) == labels).sum().item()
        tot_ex += input_ids.size(0)
    return tot_loss / tot_ex, tot_correct / tot_ex


# ------------------- Main -------------------
import math

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--bidirectional", action="store_true", default=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--min_freq", type=int, default=2)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--ckpt", type=str, default="best_lstm_agnews.pt", help="baseline checkpoint (optional but recommended)")
    args = parser.parse_args()

    print(f"[Info] Using device: {DEVICE}")
    set_seed(42)

    # 1) Load dataset
    ds = load_dataset("ag_news")
    train_ds = ds["train"]; test_ds = ds["test"]

    # 2) Vocab
    itos, stoi = build_vocab(train_ds, min_freq=args.min_freq)
    vocab_size = len(itos)
    print(f"[Info] Vocab size: {vocab_size}")

    # 3) Dataloaders
    train_loader = DataLoader(AGNewsTorch(train_ds, stoi), batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    test_loader  = DataLoader(AGNewsTorch(test_ds, stoi),  batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    # 4) Build LoRA model
    model = LSTMClassifierWithLoRA(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.layers,
        num_classes=4,
        pad_idx=PAD,
        bidirectional=args.bidirectional,
        dropout=args.dropout,
        lora_rank=args.rank,
        lora_alpha=args.alpha,
    ).to(DEVICE)

    # 5) Optionally load baseline checkpoint (embedding & lstm & fc base 会自动对上名称)
    if args.ckpt:
        try:
            sd = torch.load(args.ckpt, map_location="cpu")
            missing, unexpected = model.load_state_dict(sd, strict=False)
            print(f"[Info] Loaded baseline ckpt: {args.ckpt}")
            if missing:    print(f"[State] Missing keys: {missing}")
            if unexpected: print(f"[State] Unexpected keys: {unexpected}")
        except FileNotFoundError:
            print(f"[Warn] Baseline ckpt '{args.ckpt}' not found. You can still train, but results may be worse.")

    # 6) Only train LoRA params (A/B) + （LoRA on top 的 fc 不包含 base 权重）
    total, trainable, ratio = count_params(model)
    print(f"[Params] total={total:,}  trainable={trainable:,}  ratio={ratio:.4f}")

    # 7) Train
    criterion = nn.CrossEntropyLoss()
    # 只选择 requires_grad=True 的参数
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    best_acc = 0.0
    for ep in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        te_loss, te_acc = evaluate(model, test_loader, criterion, DEVICE)
        if te_acc > best_acc:
            best_acc = te_acc
            torch.save(model.state_dict(), "best_lstm_agnews_lora.pt")
        print(f"Epoch {ep:02d} | Train {tr_loss:.4f}/{tr_acc:.4f} | Test {te_loss:.4f}/{te_acc:.4f}")

    print(f"[Done] Best Test Acc (LoRA): {best_acc:.4f}")
    print("Saved weights to: best_lstm_agnews_lora.pt")


if __name__ == "__main__":
    main()
