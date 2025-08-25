# models.py
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils


class RNNClassifier(nn.Module):
    """
    单向 vanilla RNN 分类器。
    - 使用 nn.RNN（tanh），多层堆叠，dropout 只在层间生效（num_layers>1 时）。
    - forward 内部根据 PAD 计算长度并 pack，忽略 padding 的影响。
    - 结构：Embedding -> RNN -> LayerNorm -> Dropout -> Linear
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3,
        pad_idx: int = 1,
        emb_dropout: float = 0.2,
        rnn_dropout: float = 0.3,
        post_dropout: float = 0.3,
        nonlinearity: str = 'tanh',
        bidirectional: bool = False
    ):
        super().__init__()
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.emb_drop = nn.Dropout(emb_dropout)

        self.rnn = nn.RNN(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity=nonlinearity,
            dropout=rnn_dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.ln = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(post_dropout)
        self.fc = nn.Linear(out_dim, output_dim)

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [B, T] 的 token id 序列。
        lengths: 可选，若不给则内部按 pad_idx 自动计算；仅用于 pack，不改变输出行为。
        """
        emb = self.emb_drop(self.embedding(x))  # [B, T, E]

        if lengths is None:
            lengths = (x != self.pad_idx).sum(dim=1).cpu()
            # 避免出现 0 长度导致 pack 报错
            lengths = torch.clamp(lengths, min=1)
        else:
            lengths = lengths.cpu()

        packed = rnn_utils.pack_padded_sequence(emb, lengths, batch_first=True, enforce_sorted=False)
        _, h_n = self.rnn(packed)  # h_n: [num_layers * num_directions, B, H]
        # 取最后一层隐藏态；若为双向，拼接两个方向
        if self.rnn.bidirectional:
            h_last = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            h_last = h_n[-1]

        h_last = self.ln(h_last)
        h_last = self.dropout(h_last)
        logits = self.fc(h_last)   # [B, C]
        return logits