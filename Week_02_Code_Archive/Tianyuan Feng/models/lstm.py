import torch
import torch.nn as nn

class LSTMEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden, num_layers=1, bidirectional=False, dropout=0.2, pad_idx=0):
        super().__init__()
        # 2. 定义词嵌入层 (Embedding Layer)
        #    - 这是一个大小为 (词汇表大小 x 嵌入维度) 的查找表。
        #    - vocab_size: 词汇表中的单词总数。
        #    - emb_dim: 希望将每个单词转换成的向量的维度。
        #    - padding_idx: 一个重要的优化参数。它告诉模型，遇到值为 pad_idx 的输入时，
        #      应将其视为无效的填充部分，其对应的向量输出为全零，并且在训练中不更新。
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        # 3. 定义核心的 LSTM 层
        #    - emb_dim: 输入给LSTM的每个元素的维度（即词向量的维度）。
        #    - hidden: LSTM内部隐藏状态的维度。
        #    - num_layers: LSTM的层数（深度）。
        #    - batch_first=True: 一个非常重要的约定，表示输入数据的维度顺序是 (批次大小, 句子长度, 特征维度)。
        #    - bidirectional: 是否使用双向LSTM。
        #    - dropout: 如果 num_layers > 1，在LSTM各层之间应用的Dropout比率，用于防止过拟合。
        self.lstm = nn.LSTM(
            emb_dim, hidden, num_layers=num_layers, batch_first=True,
            bidirectional=bidirectional, dropout=(dropout if num_layers>1 else 0.0)
        )
        # 4. 定义一个 Dropout 层
        #    - 这个Dropout层将应用在LSTM的最终输出上，作为另一层正则化，防止过拟合。
        self.drop = nn.Dropout(dropout)
        # 5. 计算并存储最终输出向量的维度
        #    - 如果是单向LSTM，输出维度就是hidden的大小。
        #    - 如果是双向LSTM，最终的输出是由前向和后向的最后一个隐藏状态拼接而成的，所以维度是 hidden 的两倍。
        self.out_dim = hidden*(2 if bidirectional else 1)

    def forward(self, x, lengths):
        # 1. 嵌入 (Embedding)
        #    - 将输入的数字ID序列 `x` 传入词嵌入层 `self.emb`。
        #    - 输出 `emb` 是一个形状为 (批次大小, 句子长度, 嵌入维度) 的词向量序列。
        emb = self.emb(x)
        # 2. 打包序列 (Packing)
        #    - 这是一个关键的性能优化步骤，用于处理一个批次中长短不一的句子。
        #    - 它将填充过的（padded）词向量序列 `emb` 和每个句子的真实长度 `lengths` “打包”成一个更紧凑的对象 `packed`。
        #    - 这样做可以告诉LSTM在处理每个句子时，只需要计算到其真实长度即可，忽略掉后面的填充部分，从而节省计算资源并可能提高模型性能。
        #    - .cpu() 是因为这个打包操作要求长度张量在CPU上。
        packed = torch.nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        # 3. LSTM计算
        #    - 将打包好的数据送入 `self.lstm` 层。
        #    - LSTM层会返回两个主要部分：`output` (所有时间步的输出) 和 `(h, c)` (最后一个时间步的隐藏状态和细胞状态)。
        #    - 我们通常只需要最后一个时间步的隐藏状态 `h` 来代表整个句子，所以用 `_` 忽略其他输出。
        _, (h, _) = self.lstm(packed)
         # 4. 提取最终的句子向量
        #    - `h` 的形状是 [层数 * 方向数, 批次大小, 隐藏维度]。
        #    - 如果是单向模型，我们只需要最后一层的隐藏状态，即 `h[-1]`。
        #    - 如果是双向模型，最后一层的隐藏状态包含了前向 `h[-2]` 和后向 `h[-1]` 两个部分。
        #      我们将这两个向量沿维度1（特征维度）进行拼接(concatenate)，形成一个更丰富的句子表示。
        h_last = torch.cat([h[-2], h[-1]], dim=1) if self.lstm.bidirectional else h[-1]
        # 5. 返回结果
        #    - 将最终提取出的句子向量通过一个Dropout层进行正则化，然后返回。
        return self.drop(h_last)
