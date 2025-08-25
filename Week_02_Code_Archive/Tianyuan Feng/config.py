# 导入dataclasses模块中的dataclass装饰器。
# 这个装饰器可以自动为类生成一些基础方法（如构造函数__init__），让创建数据存储类变得非常简洁。

from dataclasses import dataclass

@dataclass
class TrainConfig:
    # --- 数据相关参数 ---
    max_len: int = 256              # 输入句子的最大长度（超过则截断，不足则填充）。
    vocab_size: int = 30000         # 词汇表的最大容量（只保留最常见的3万个词）。
    min_freq: int = 2               # 一个词在训练集中最少出现的次数，低于此频率的词将被忽略。
    batch_size: int = 64            # 训练时，每批投入模型的样本数量。
    num_workers: int = 0            # 加载数据时使用的子进程数量（0表示在主进程中加载）。
    val_ratio: float = 0.1          # 从原始训练集中划分出10%作为验证集。
    eval_batch_size: int = 128      # 在验证和测试时，每批的样本数量（因为没有反向传播，通常可以设置得更大以加快速度）。

    # --- 模型相关参数 ---
    model_type: str = "lstm"        # 模型的类型，可以是 "lstm" 或 "lstm_lora"。
    emb_dim: int = 256              # 词嵌入向量的维度（每个词转换成的数字向量的长度）。
    hidden: int = 256               # LSTM隐藏层的维度。
    layers: int = 2                 # LSTM的层数（深度）。
    bidirectional: bool = True      # 是否使用双向LSTM。
    dropout: float = 0.5            # Dropout比率，用于防止模型过拟合。
    
    # --- LoRA微调相关参数 ---
    base_model_path: str = None     # 用于LoRA微调的基座模型权重文件路径，None表示不加载。

    # --- LoRA专属参数 ---
    lora_r: int = 8                 # LoRA的秩（rank），决定了适配器的大小，越小参数量越少。
    lora_alpha: int = 16            # LoRA的缩放因子。
    lora_dropout: float = 0.05      # LoRA模块内部的Dropout比率。
    freeze_base: bool = False       # 是否冻结基座模型（进行参数高效微调的关键）。
    freeze_embed: bool = False      # 是否冻結词嵌入层。
    merge_lora: bool = False        # 是否在推理前将LoRA权重合并到基座中。
    save_lora_only: bool = False    # 是否只保存LoRA的适配器权重。

    # --- 训练过程相关参数 ---
    epochs: int = 6                 # 训练的总轮数。
    lr: float = 1e-3                # 学习率（Learning Rate），模型参数更新的步长。
    weight_decay: float = 1e-2      # 权重衰减，一种正则化手段，防止权重过大。
    grad_clip: float = 1.0          # 梯度裁剪的阈值，防止梯度爆炸。

    # --- 评估流程控制 ---
    do_eval: bool = True            # 是否在训练后进行最终的测试评估。
    save_best: bool = True          # 是否根据验证集损失只保存表现最好的模型。

    # --- 其他杂项 ---
    seed: int = 42                  # 随机种子，用于保证实验结果的可复现性。
    artifacts_dir: str = "artifacts" # 保存模型、词汇表等输出文件的目录名。

    # --- 编排（Orchestration） ---
    run_both: bool = False          # 一个遗留参数，用于控制是否连续运行两种模型。