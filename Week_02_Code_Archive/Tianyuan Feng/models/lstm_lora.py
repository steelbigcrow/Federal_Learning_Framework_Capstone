import torch
import torch.nn as nn
import loralib as lora

class LSTMLoRACell(nn.Module):
    def __init__(self, input_size, hidden_size, lora_r=8, lora_alpha=16, lora_dropout=0.05,
                 freeze_base=True, merge_weights=False):
        # 1. 调用父类nn.Module的构造函数，这是PyTorch模块的标准写法。
        super().__init__()
        # 2. 存储隐藏层的大小
        self.hidden_size = hidden_size
        # 3. 创建 input-to-hidden 的线性变换层。
        #    - 它是一个 lora.Linear 层，内部包含了被冻结的“基座”权重和可训练的“适配器”。
        #    - `4 * hidden_size` 是一个优化技巧，因为LSTM有4个门，我们可以通过一次大的矩阵运算，同时计算出所有门的结果，然后再切分开。
        self.proj_ih = lora.Linear(
            input_size, 4 * hidden_size, r=lora_r, lora_alpha=lora_alpha, 
            lora_dropout=lora_dropout, merge_weights=merge_weights
        )
        # 4. 创建 hidden-to-hidden 的线性变换层。
        #    - 它同样是一个 lora.Linear 层。
        self.proj_hh = lora.Linear(
            hidden_size, 4 * hidden_size, r=lora_r, lora_alpha=lora_alpha, 
            lora_dropout=lora_dropout, merge_weights=merge_weights
        )
        # 5. 如果 freeze_base 为 True，则冻结所有 lora.Linear 层中的基座权重。
        if freeze_base:
            lora.mark_only_lora_as_trainable(self, bias='none')
        # 6. 对基座权重进行随机初始化。
        #    - 这是一个标准的初始化流程，确保模型在开始训练前有一个合理的初始状态。
        stdv = 1.0 / (hidden_size ** 0.5)
        with torch.no_grad():
            # 在这个代码块中，不进行梯度计算，以提高效率。
            for proj in [self.proj_ih, self.proj_hh]:
                # `weight`属性指向基座权重，用均匀分布的随机数填充它。
                proj.weight.uniform_(-stdv, stdv)
                if proj.bias is not None:
                    # 同样初始化偏置项。
                    proj.bias.uniform_(-stdv, stdv)

    def forward(self, x_t, hx):
        # 1. 接收当前时间步的输入 `x_t` 和上一个时间步的状态 `hx`。
        #`h_prev`: 上一步的短期工作记忆（隐藏状态）。
        #`c_prev`: 上一步的长期记忆（细胞状态）。
        h_prev, c_prev = hx
        # 2. 计算所有“门”的总输入。
        # 这是当前输入 `x_t` 和上一步记忆 `h_prev` 共同作用的结果。
        gates = self.proj_ih(x_t) + self.proj_hh(h_prev)
        # 3. 将结果切分成4份，分别供给4个门。
        i, f, g, o = gates.chunk(4, dim=1)
        # 4. LSTM核心的“记忆管理”数学逻辑。
        #    a) 遗忘门 (Forget Gate) `f`: 决定“遗忘”多少旧的长期记忆。
        #       - `torch.sigmoid(f)` 将 `f` 的值压缩到0-1之间，作为遗忘比例。
        #       - `f * c_prev`：将旧的长期记忆 `c_prev` 乘以这个比例。
        #    b) 输入门 (Input Gate) `i` 与 候选门 `g`: 决定“输入”多少新信息。
        #       - `torch.tanh(g)` 从当前输入中提炼出一个“-1到1”的“候选记忆”。
        #       - `torch.sigmoid(i)` 决定这个“候选记忆”有多大价值，应该采纳多少。
        #    c) 更新长期记忆 `c_t`: 新的长期记忆 = (被保留的旧记忆) + (被采纳的新记忆)。
        c_t = torch.sigmoid(f) * c_prev + torch.sigmoid(i) * torch.tanh(g)
        # 5. 输出门 (Output Gate) `o`: 决定从新的长期记忆中，输出什么作为当前的“短期工作记忆”。
        #    - `torch.sigmoid(o)` 决定哪些信息可以“对外输出”。
        #    - `torch.tanh(c_t)` 再次处理长期记忆，使其值在-1到1之间。
        #    - 两者相乘，得到最终的短期记忆 `h_t`。
        h_t = torch.sigmoid(o) * torch.tanh(c_t)
        # 6. 返回新的短期记忆和长期记忆，供下一个时间步使用。
        return h_t, c_t

class LSTMLoRAEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden, num_layers=1, bidirectional=False, dropout=0.2, pad_idx=0,
                 lora_r=8, lora_alpha=16, lora_dropout=0.05, freeze_base=True, merge_weights=False):
        # 1. 调用父类构造函数。
        super().__init__()
        # 2. 创建词嵌入层。将单词ID转化为词向量。
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        # 3. 创建Dropout层
        self.drop = nn.Dropout(dropout)
        # 4. 存储基本配置。
        self.hidden = hidden
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        # 5. 创建一个 nn.ModuleList 来存放所有的LSTMLoRACell。
        self.cells = nn.ModuleList()
        # 6. 循环创建每一层的Cell。
        for l in range(num_layers):
            # 7. 确定当前层Cell的输入维度。
            #    - 第0层直接接收词向量，输入是 `emb_dim`。
            #    - 后续层接收上一层的输出，如果是双向，维度是 `hidden * 2`，否则是 `hidden`。
            input_dim = emb_dim if l == 0 else hidden * (2 if bidirectional else 1)
            # 8. 创建前向传播的Cell并添加到列表中。
            self.cells.append(LSTMLoRACell(input_dim, hidden, lora_r, lora_alpha, lora_dropout, freeze_base, merge_weights))
            # 9. 如果是双向lstm，再创建一个后向传播的Cell。
            if bidirectional:
                self.cells.append(LSTMLoRACell(input_dim, hidden, lora_r, lora_alpha, lora_dropout, freeze_base, merge_weights))
        # 10. 计算并存储维度。
        self.out_dim = hidden * (2 if bidirectional else 1)

    def forward(self, x, lengths):
        # 1. 获取输入尺寸：B=批大小, T=句子长度。
        B, T = x.size()
        # 2. 将输入的ID序列通过“炼油厂”和“泄压阀”，得到处理好的词向量序列 `h_in`。
        h_in = self.drop(self.emb(x))

        for l in range(self.num_layers):
            # 4.如果是双向lstm
            if self.bidirectional:
                 # 5. 从列表中获取这一层的前向和后向Cell。
                f_cell = self.cells[l * 2]
                b_cell = self.cells[l * 2 + 1]
                # 6. 初始化前向的初始状态（短期和长期记忆都为0）。
                h_t_f = torch.zeros(B, self.hidden, device=x.device)
                c_t_f = torch.zeros(B, self.hidden, device=x.device)
                f_outs = []
                # 7. 从t=0到结尾，逐个单词处理。
                for t in range(T):
                    h_t_f, c_t_f = f_cell(h_in[:, t, :], (h_t_f, c_t_f))
                    f_outs.append(h_t_f.unsqueeze(1))
                h_f = torch.cat(f_outs, dim=1)
                 # 8. 初始化后向的初始状态。
                h_t_b = torch.zeros(B, self.hidden, device=x.device)
                c_t_b = torch.zeros(B, self.hidden, device=x.device)
                b_outs = []
                # 9. 从t=T-1到0，反向逐个单词处理。
                for t in reversed(range(T)):
                    h_t_b, c_t_b = b_cell(h_in[:, t, :], (h_t_b, c_t_b))
                    b_outs.insert(0, h_t_b.unsqueeze(1))
                h_b = torch.cat(b_outs, dim=1)
                # 10. 将前向和后向拼接起来，作为下一层的输入。
                h_in = torch.cat([h_f, h_b], dim=2)
            # 11. 如果是单向...
            else:
                cell = self.cells[l]
                h_t = torch.zeros(B, self.hidden, device=x.device)
                c_t = torch.zeros(B, self.hidden, device=x.device)
                uni_outs = []
                 # 12. 只进行一次前向冲程。
                for t in range(T):
                    h_t, c_t = cell(h_in[:, t, :], (h_t, c_t))
                    uni_outs.append(h_t.unsqueeze(1))
                h_in = torch.cat(uni_outs, dim=1)
            # 13. 如果不是最后一层，则对输出使用Dropout，准备送入下一层。
            if l < self.num_layers - 1:
                h_in = self.drop(h_in)
        # 14. 提取最终的句子表示向量。
        # 15. `(lengths - 1)`: 获取每个句子的最后一个有效单词的索引。
        # 16. `.view(...).expand(...)`: 将索引的形状调整为与输出张量匹配，以便进行索引。
        # 17. `.gather(1, idx)`: 一个高级索引操作，根据我们计算出的索引，精确地从所有时间步的输出中“抓取”每个句子对应的最后一个有效输出。
        idx = (lengths - 1).clamp(min=0).view(-1, 1, 1).expand(B, 1, self.out_dim)
        last_h = h_in.gather(1, idx).squeeze(1)
        return last_h