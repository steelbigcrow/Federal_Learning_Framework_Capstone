import re
import random
from collections import Counter
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_device()
print(f"[Info] Using device: {DEVICE}")

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# ---------- 1) 数据集：HuggingFace datasets ----------
ds = load_dataset("ag_news")
train_ds = ds["train"]
test_ds  = ds["test"]

# ---------- 2) 简单英文分词 & 词表 ----------
def tokenize(text: str) -> List[str]:
    # 小写 + 基本的单词/标点切分
    return re.findall(r"\w+|[^\w\s]", text.lower())

def yield_tokens(dataset_split):
    for ex in dataset_split:
        # AG News里文本字段是 'text'
        yield tokenize(ex["text"])

SPECIALS = ["<pad>", "<unk>"]
PAD, UNK = 0, 1

def build_vocab(dataset_split, min_freq=2):
    counter = Counter()
    for tokens in yield_tokens(dataset_split):
        counter.update(tokens)
    itos = SPECIALS[:]  # index -> token
    for tok, freq in counter.items():
        if freq >= min_freq:
            itos.append(tok)
    stoi = {tok: i for i, tok in enumerate(itos)}  # token -> index
    return itos, stoi

itos, stoi = build_vocab(train_ds, min_freq=2)
vocab_size = len(itos)
print(f"[Info] Vocab size: {vocab_size}")

def encode(text: str) -> List[int]:
    return [stoi.get(t, UNK) for t in tokenize(text)]

def label_encode(label: int) -> int:
    # HF 的 ag_news label 已经是 0..3
    return int(label)

# ---------- 3) PyTorch Dataset + collate（动态 padding） ----------
class AGNewsTorch(Dataset):
    def __init__(self, hf_split):
        self.data = hf_split

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        ids = encode(ex["text"])
        y = label_encode(ex["label"])
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

train_loader = DataLoader(AGNewsTorch(train_ds), batch_size=64, shuffle=True, collate_fn=collate_batch)
test_loader  = DataLoader(AGNewsTorch(test_ds),  batch_size=64, shuffle=False, collate_fn=collate_batch)

# ---------- 4) LSTM 模型 ----------
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, num_classes, pad_idx, bidirectional=True, dropout=0.4):
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

    def forward(self, input_ids, lengths):
        emb = self.embedding(input_ids)
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        if self.lstm.bidirectional:
            h_fw = h_n[-2, :, :]
            h_bw = h_n[-1, :, :]
            feat = self.dropout(torch.cat([h_fw, h_bw], dim=1))
        else:
            feat = self.dropout(h_n[-1, :, :])
        return self.fc(feat)

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    tot_loss, tot_correct, tot_ex = 0.0, 0, 0
    for input_ids, lengths, labels in loader:
        input_ids, lengths, labels = input_ids.to(DEVICE), lengths.to(DEVICE), labels.to(DEVICE)
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
def evaluate(model, loader, criterion):
    model.eval()
    tot_loss, tot_correct, tot_ex = 0.0, 0, 0
    for input_ids, lengths, labels in loader:
        input_ids, lengths, labels = input_ids.to(DEVICE), lengths.to(DEVICE), labels.to(DEVICE)
        logits = model(input_ids, lengths)
        loss = criterion(logits, labels)
        tot_loss += loss.item() * input_ids.size(0)
        tot_correct += (logits.argmax(1) == labels).sum().item()
        tot_ex += input_ids.size(0)
    return tot_loss / tot_ex, tot_correct / tot_ex

def main():
    embed_dim = 128
    hidden_dim = 256
    num_layers = 2
    num_classes = 4
    num_epochs = 3
    lr = 2e-3

    model = LSTMClassifier(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=num_classes,
        pad_idx=PAD,
        bidirectional=True,
        dropout=0.4,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_acc = 0.0
    for ep in range(1, num_epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        te_loss, te_acc = evaluate(model, test_loader, criterion)
        if te_acc > best_acc:
            best_acc = te_acc
            torch.save(model.state_dict(), "best_lstm_agnews.pt")
        print(f"Epoch {ep:02d} | Train {tr_loss:.4f}/{tr_acc:.4f} | Test {te_loss:.4f}/{te_acc:.4f}")

    print(f"[Done] Best Test Acc: {best_acc:.4f}")

if __name__ == "__main__":
    main()
