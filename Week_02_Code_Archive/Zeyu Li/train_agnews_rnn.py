import os
import re
import math
import random
from collections import Counter
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from tqdm import tqdm

# ---------------------------
# 超参数（4GB 显存安全配置）
# ---------------------------
MAX_LEN = 128# 文本最大长度
MIN_FREQ = 3# 词表最小频率
EMB_DIM = 100# 词向量维度
HID_DIM = 128# 隐藏层维度
NUM_LAYERS = 1# LSTM 层数
DROPOUT = 0.2# LSTM dropout
BATCH_SIZE = 128# 批大小
LR = 2e-3# 学习率
EPOCHS = 6# 训练轮数
SEED = 42# 随机种子

random.seed(SEED)
torch.manual_seed(SEED)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True  # 长度固定时能加速
#Reproducibility is ensured by setting random seeds for random and torch . The script also
#determines the device ( cuda if available, otherwise cpu ) and enables
#torch.backends.cudnn.benchmark for potential speedup when input lengths are fixed.


def simple_tokenize(text: str) -> List[str]:
    # 非字母数字转空格，转小写，按空白切分
    text = re.sub(r"[^A-Za-z0-9]+", " ", text).lower()
    return text.strip().split()

# ---------------------------
# 1) 加载数据
# ---------------------------
ds = load_dataset("ag_news")  # train/test splits；label: 0..3
NUM_CLASSES = 4

# ---------------------------
# 2) 构建词表
# ---------------------------
counter = Counter()
for ex in ds["train"]:
    counter.update(simple_tokenize(ex["text"]))

itos = ["<pad>", "<unk>"]
for tok, freq in counter.items():
    if freq >= MIN_FREQ:
        itos.append(tok)# 只保留频率 >= MIN_FREQ 的词
stoi = {tok: i for i, tok in enumerate(itos)}
PAD_IDX = stoi["<pad>"]# padding token index
UNK_IDX = stoi["<unk>"]# unknown token index
VOCAB_SIZE = len(itos)# 词表大小
print(f"Vocab size: {VOCAB_SIZE}")# 输出词表大小

def encode(text: str, max_len: int = MAX_LEN) -> Tuple[torch.Tensor, int]:
    toks = simple_tokenize(text)
    ids = [stoi.get(t, UNK_IDX) for t in toks][:max_len]
    length = len(ids)
    if length < max_len:
        ids += [PAD_IDX] * (max_len - length)
    return torch.tensor(ids, dtype=torch.long), length

class AGNewsDataset(Dataset):
    def __init__(self, split):
        self.texts = split["text"]
        self.labels = split["label"]
    def __len__(self): return len(self.labels)# 返回数据集大小
    def __getitem__(self, idx):# 获取单个样本
        x, l = encode(self.texts[idx])
        y = int(self.labels[idx])
        return x, l, y

train_data = AGNewsDataset(ds["train"])
test_data  = AGNewsDataset(ds["test"])

def collate_fn(batch):# 自定义 collate_fn 处理变长序列
    xs, ls, ys = zip(*batch)
    return (torch.stack(xs, 0),
            torch.tensor(ls, dtype=torch.long),
            torch.tensor(ys, dtype=torch.long))

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, collate_fn=collate_fn)
test_loader  = DataLoader(test_data,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, collate_fn=collate_fn)

# ---------------------------
# 3) 定义模型（双向 LSTM）
# ---------------------------
class TextBiLSTM(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, num_layers, num_classes,
                 pad_idx, dropout=0.2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hid_dim * 2, num_classes)

    def forward(self, x, lengths):
        # x: [B, T]
        emb = self.emb(x)  # [B, T, E]
        # 使用 pack_padded_sequence 让 LSTM 忽略 PAD
        lengths = lengths.clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(),
                                                   batch_first=True, enforce_sorted=False)
        packed_out, (h_n, _) = self.lstm(packed)
        # h_n: [num_layers*2, B, H]  -> 取最后一层的正/反向隐状态做拼接
        h_last = torch.cat([h_n[-2], h_n[-1]], dim=1)  # [B, 2H]
        out = self.dropout(h_last)
        return self.fc(out)

model = TextBiLSTM(VOCAB_SIZE, EMB_DIM, HID_DIM, NUM_LAYERS, NUM_CLASSES,
                   PAD_IDX, DROPOUT).to(device)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

# ---------------------------
# 4) 训练 & 验证
# ---------------------------
def evaluate(loader):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
        for x, l, y in loader:
            x, l, y = x.to(device), l.to(device), y.to(device)
            logits = model(x, l)
            loss = criterion(logits, y)
            loss_sum += loss.item() * y.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total   += y.size(0)
    return loss_sum / total, correct / total

best_acc = 0.0
for epoch in range(1, EPOCHS + 1):
    model.train()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
    for x, l, y in pbar:
        x, l, y = x.to(device), l.to(device), y.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            logits = model(x, l)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    val_loss, val_acc = evaluate(test_loader)
    print(f"[Eval] loss={val_loss:.4f} acc={val_acc:.4%}")
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "agnews_rnn_best.pt")
        print(f"  ✓ New best acc {best_acc:.4%}, model saved -> agnews_rnn_best.pt")

print("Training done. Best acc:", f"{best_acc:.4%}")
