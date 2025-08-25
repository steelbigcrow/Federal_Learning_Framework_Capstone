# data.py
import html
import re
from collections import Counter
from typing import List, Tuple, Dict, Callable, Iterable

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.utils.rnn as rnn_utils

try:
    from nltk.tokenize import word_tokenize as _nltk_word_tokenize  # type: ignore
    _HAS_NLTK = True
except Exception:
    _HAS_NLTK = False


# ==========================
# 1) 分词与词表
# ==========================
def tokenize(text: str) -> List[str]:
    """
    简单小写 + HTML 清洗 + 轻度规范化 + NLTK 分词（不可用时降级）。
    """
    text = text.lower()
    text = re.sub(r'<br\s*/?>', ' ', text)        # 去掉 <br/>
    text = re.sub(r'<.*?>', ' ', text)            # 去掉其他 HTML 标记
    text = html.unescape(text)                    # 反转义
    text = re.sub(r'\d+', ' <num> ', text)        # 数字归一化
    text = re.sub(r"[^a-z0-9'!?.,;:()\-\s]+", ' ', text)  # 清掉奇异字符
    text = re.sub(r'\s+', ' ', text).strip()

    if _HAS_NLTK:
        try:
            return _nltk_word_tokenize(text)
        except LookupError:
            # 自动尝试下载 punkt
            try:
                import nltk  # type: ignore
                nltk.download('punkt', quiet=True)
                return _nltk_word_tokenize(text)
            except Exception:
                pass

    # 回退：简单正则分词（保留标点）
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?|[!?.,;:()\-]", text)


def build_vocab(
    texts: Iterable[str],
    min_freq: int = 3,
    max_vocab_size: int = 30000,
    specials: Tuple[str, str] = ('<unk>', '<pad>')
) -> Dict[str, int]:
    """
    统计词频并构建词表。
    默认保留 specials：('<unk>', '<pad>') => id 分别为 0、1。
    """
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))

    counter = Counter({w: c for w, c in counter.items() if c >= min_freq})
    most_common = counter.most_common(max_vocab_size)

    vocab = {specials[0]: 0, specials[1]: 1}  # <unk>=0, <pad>=1
    for idx, (word, _) in enumerate(most_common, start=len(vocab)):
        vocab[word] = idx
    return vocab


# ==========================
# 2) Dataset 与 Collate
# ==========================
class IMDBDataset(Dataset):
    """
    文本->id 的数据集。按 max_len 截断。
    """
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        vocab: Dict[str, int],
        max_len: int = 200,
        unk_token: str = '<unk>'
    ):
        assert len(texts) == len(labels), "texts 与 labels 数量不一致"
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len
        self.unk_idx = vocab.get(unk_token, 0)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        tokens = tokenize(self.texts[idx])
        ids = [self.vocab.get(tok, self.unk_idx) for tok in tokens]
        ids = ids[: self.max_len]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)


class CollateWithPad:
    """
    顶层可调用类：可被 Windows 多进程 DataLoader 安全 pickle。
    动态 padding，并返回序列长度（供 pack 使用或调试）。
    """
    def __init__(self, pad_idx: int):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        texts, labels = zip(*batch)
        lengths = torch.tensor([len(t) for t in texts], dtype=torch.long)
        padded_texts = rnn_utils.pad_sequence(
            texts, batch_first=True, padding_value=self.pad_idx
        )
        labels_t = torch.stack(labels)
        return padded_texts, labels_t, lengths


def get_collate_fn(pad_idx: int) -> Callable:
    """
    返回使用指定 pad_idx 的可序列化 collate_fn。
    """
    return CollateWithPad(pad_idx)


# ==========================
# 3) DataLoader 构建工具
# ==========================
def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    pad_idx: int,
    shuffle: bool = True,
    num_workers: int = 2,
    pin_memory: bool = True,
    persistent_workers: bool = True
) -> DataLoader:
    """
    用指定 pad_idx 的 collate_fn 构建 DataLoader。
    """
    collate_fn = get_collate_fn(pad_idx)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
    )


def create_datasets(
    X_train: List[str],
    y_train: List[int],
    X_val: List[str],
    y_val: List[int],
    vocab: Dict[str, int],
    max_len: int = 200
) -> Tuple[IMDBDataset, IMDBDataset, int, int]:
    """
    基于给定划分创建训练/验证数据集，并返回 pad/unk 索引。
    """
    train_ds = IMDBDataset(X_train, y_train, vocab, max_len=max_len)
    val_ds = IMDBDataset(X_val, y_val, vocab, max_len=max_len)
    pad_idx = vocab.get('<pad>', 1)
    unk_idx = vocab.get('<unk>', 0)
    return train_ds, val_ds, pad_idx, unk_idx