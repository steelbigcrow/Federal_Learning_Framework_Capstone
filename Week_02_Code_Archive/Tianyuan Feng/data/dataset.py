import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from .vocab import Vocab
from .utils import encode

class TextClsDataset(Dataset):
    # 它接收所有文本(texts)、所有标签(labels)、词汇表(vocab)和句子的最大长度(max_len)
    def __init__(self, texts, labels, vocab: Vocab, max_len: int):
        self.texts, self.labels = texts, labels
        self.vocab, self.max_len = vocab, max_len

    # 返回数据集中样本的总数量。
    def __len__(self): return len(self.texts)
    # 这是Dataset类最核心的方法。它定义了如何根据一个索引(idx)来获取并处理单个数据。
    def __getitem__(self, idx):
        # a) 调用我们之前在 utils.py 中定义的 encode 函数。
        #    - 这个函数负责将原始的文本字符串，转换成一个固定长度的、由单词ID组成的数字列表
        ids, L = encode(self.texts[idx], self.vocab, self.max_len)
        # b) 将处理好的数据转换为PyTorch的Tensor格式并返回。
        #    - Tensor是PyTorch中进行计算的基本数据结构。
        #    - 返回内容包括：数字ID列表、句子的真实长度、以及对应的标签。
        return torch.tensor(ids, dtype=torch.long), torch.tensor(L, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

class IMDBDataModule:
    def __init__(self, cfg):
        #    - 它接收包含了所有配置的 cfg 对象。
        self.cfg = cfg
        self.vocab = None
        self.train_loader = self.val_loader = self.test_loader = None

#    - 这是数据模块的核心，执行所有数据准备步骤。
    def prepare(self):
        # a) 从Hugging Face Hub下载IMDB数据集。
        ds = load_dataset("imdb")
        # b) 提取所有原始训练文本和标签。
        all_train_texts = [ex["text"] for ex in ds["train"]]
        all_train_labels = [ex["label"] for ex in ds["train"]]
        # c) 基于所有训练文本，构建词汇表(Vocab)。
        self.vocab = Vocab.build(all_train_texts, max_size=self.cfg.vocab_size, min_freq=self.cfg.min_freq)
        from os import makedirs, path
        makedirs(self.cfg.artifacts_dir, exist_ok=True)
        # d) 将构建好的词汇表保存为JSON文件，以备后用。
        self.vocab.to_json(path.join(self.cfg.artifacts_dir, "vocab.json"))
        # e) 使用sklearn库的train_test_split函数，将训练数据分割为训练集和验证集。
        #    - test_size=self.cfg.val_ratio: 定义了验证集所占的比例。
        #    - random_state=self.cfg.seed: 确保每次分割的结果都相同，便于实验复现。
        #    - stratify=all_train_labels: 保证分割后的训练集和验证集中，正面/负面评论的比例与原始数据一致。
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            all_train_texts, all_train_labels, test_size=self.cfg.val_ratio, 
            random_state=self.cfg.seed, stratify=all_train_labels
        )
        # f) 检查是否有可用的GPU，这会影响数据加载的效率。
        pin = torch.cuda.is_available()
        # g) 为分割好的三组数据（训练、验证、测试）分别创建“配餐员”(TextClsDataset实例)。
        train_ds = TextClsDataset(train_texts, train_labels, self.vocab, self.cfg.max_len)
        val_ds = TextClsDataset(val_texts, val_labels, self.vocab, self.cfg.max_len)
        test_ds = TextClsDataset([ex["text"] for ex in ds["test"]], [ex["label"] for ex in ds["test"]], self.vocab, self.cfg.max_len)
        # h) 创建三个DataLoader，分别负责运送训练、验证和测试的batch。
        #    - DataLoader会自动将准备好的单份数据，打包成设定大小(batch_size)的批次。
        #    - shuffle=True: 训练传送带会打乱餐包的顺序，防止模型按固定顺序学习产生偏见。
        #    - pin_memory=pin: 如果有GPU，这个选项可以加速数据从CPU到GPU的传输。
        self.train_loader = DataLoader(train_ds, batch_size=self.cfg.batch_size, shuffle=True, num_workers=self.cfg.num_workers, pin_memory=pin)
        self.val_loader = DataLoader(val_ds, batch_size=self.cfg.eval_batch_size, shuffle=False, num_workers=self.cfg.num_workers, pin_memory=pin)
        self.test_loader = DataLoader(test_ds, batch_size=self.cfg.eval_batch_size, shuffle=False, num_workers=self.cfg.num_workers, pin_memory=pin)
        return self