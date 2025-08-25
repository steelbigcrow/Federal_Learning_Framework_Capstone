import argparse, copy, torch
from config import TrainConfig
from data.dataset import IMDBDataModule
from data.utils import set_seed
from models.factory import ModelFactory
from trainer.trainer import Trainer

def build_argparser():
    # 1. 创建一个 ArgumentParser 对象，这是 argparse 库的核心。
    p = argparse.ArgumentParser(description="Train and finetune LSTM models for sentiment analysis.")
    
    # 首先，它创建了一个 TrainConfig 的默认实例。
    defaults = TrainConfig()
    #    - 然后，它遍历 TrainConfig 中的每一个字段（比如 'epochs', 'lr' 等）。
    for field in defaults.__dataclass_fields__:
        #    - 获取该字段的类型（比如 int, float, bool）。
        field_type = defaults.__dataclass_fields__[field].type
        #    - 如果是布尔类型，就创建一个开关式的参数（比如 --bidirectional）。
        if field_type == bool:
            p.add_argument(f"--{field}", action="store_true", default=defaults.__getattribute__(field))
            #    - 否则，就创建一个需要接收值的参数（比如 --epochs 10）。
        else:
            p.add_argument(f"--{field}", type=field_type, default=defaults.__getattribute__(field))
    return p

def pick_device():
    # 1. 检查是否有NVIDIA的GPU（通过CUDA）可用。
    if torch.cuda.is_available():
        return torch.device("cuda")
    # 2. 如果没有，再检查是否有苹果M系列芯片的GPU（通过MPS）可用。
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    # 3. 如果两者都没有，则抛出一个错误，因为本项目不支持在CPU上训练。
    else:
        raise RuntimeError("No CUDA GPU or Apple MPS available. CPU is not supported.")

def train_once(cfg: TrainConfig, device):
    # 1. 准备数据：创建 IMDBDataModule 实例并调用 .prepare() 方法来下载、处理、打包所有数据。
    dm = IMDBDataModule(cfg).prepare()
    vocab = dm.vocab
    # 2. 获取数据加载器：从准备好的数据模块中，直接获取训练、验证和测试的数据加载器。
    train_loader = dm.train_loader
    val_loader = dm.val_loader
    test_loader = dm.test_loader
    # 3. 构建模型：调用模型工厂的 .build() 方法，根据 cfg 的配置来创建相应的模型。
    model, tag = ModelFactory.build(cfg, vocab)
    # 4. 初始化训练器：创建一个 Trainer 类的实例，并把配置 cfg 和设备 device 交给它。
    trainer = Trainer(cfg, device)
    # 5. 执行训练：调用训练器的 .fit() 方法，传入模型和数据，开始核心的训练与验证循环。
    trained_model = trainer.fit(model, train_loader, val_loader)
    # 6. (可选) 执行最终评估：如果配置要求（do_eval=True），则调用 .test() 方法在测试集上进行最终评估。
    if cfg.do_eval:
        print("\n--- Starting Final Evaluation on Test Set ---")
        trainer.test(trained_model, test_loader)
    # 7. 导出成果：调用 .export() 方法，将最终训练好的模型保存到文件中。
    trainer.export(trained_model, tag)

def main():
    # 1. 创建参数解析器。
    parser = build_argparser()
    # 2. 解析从命令行（或launch.json）传入的实际参数。
    args = parser.parse_args()
    # 3. 创建最终的配置对象 `cfg`。
    #    - `vars(args)` 将解析出的参数转换为字典。
    #    - `**` 操作符将这个字典解包，作为关键字参数传递给 TrainConfig 的构造函数。
    #    - 这就实现了用命令行参数来覆盖 config.py 中的默认值。
    cfg = TrainConfig(**vars(args))
    # 4. 选择设备。
    device = pick_device()
    print("Device:", device)
    # 5. 设置随机种子以保证实验可复现。
    set_seed(cfg.seed)
    # 6. 调用核心的单次训练流程。
    train_once(cfg, device)

if __name__ == "__main__":
    main()