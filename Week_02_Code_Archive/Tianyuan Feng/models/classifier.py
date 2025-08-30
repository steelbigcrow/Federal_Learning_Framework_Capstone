import torch.nn as nn

class TextClassifier(nn.Module):
    # 构造函数
    # 它接收四个参数：
    #   - encoder: 任何一个编码器模块 (比如 LSTMEncoder)。
    #   - out_dim: 编码器输出的特征向量的维度。
    #   - num_labels: 最终需要输出的类别数量（对于情感二分类，是1）。
    #   - dropout: 应用在最终决策前的Dropout比率。
    def __init__(self, encoder: nn.Module, out_dim: int, num_labels: int = 1, dropout=0.2):
        # 1. 调用父类nn.Module的构造函数，这是PyTorch模块的标准写法。
        super().__init__()
        # 2. 将传入的编码器(encoder)保存为类的一个属性。
        #    这个编码器是模型的核心部分，负责理解文本。
        self.encoder = encoder
        # 3. 定义分类“头部”(head)。
        #    - nn.Sequential 是一个容器，它会按顺序执行放入其中的模块。
        #    - nn.Dropout(dropout): 一个正则化层。在训练时，它会以一定的概率(dropout)随机地将输入的一些元素置为零，
        #      这有助于防止模型过拟合，提高泛化能力。
        #    - nn.Linear(out_dim, num_labels): 一个全连接线性层。这是最终的决策层。
        #      它接收来自编码器的、维度为 out_dim 的特征向量，并通过一次线性变换，
        #      将其映射到维度为 num_labels 的最终输出分数(logits)。
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(out_dim, num_labels))
    def forward(self, x, lengths):
        # 1. 调用编码器
        #    - 将输入的数字ID序列 `x` 和每个序列的真实长度 `lengths` 传递给 self.encoder。
        #    - 编码器会完成所有复杂的文本理解工作，并返回一个浓缩了整句话核心语义的特征向量 `feats`。
        feats = self.encoder(x, lengths)
        # 2. 调用分类头
        #    - 将编码器输出的特征向量 `feats` 传递给 self.head。
        #    - self.head 会先对特征向量应用Dropout，然后通过线性层进行最终的分类变换。
        #    - 返回的结果是模型的原始预测分数(logits)，这个分数随后将被用于计算损失或生成最终的预测。
        return self.head(feats)
