import torch
from .lstm import LSTMEncoder
from .lstm_lora import LSTMLoRAEncoder
from .classifier import TextClassifier

class ModelFactory:
    @staticmethod
    def build(cfg, vocab):
        # 检查配置中指定的模型类型是否为 "lstm"
        if cfg.model_type == "lstm":
            print("Using model: LSTM (standard)")
            # 1. 实例化 LSTMEncoder：从 lstm.py 中创建标准LSTM编码器实例。
            #    将词汇表大小、嵌入维度、隐藏层大小等所有需要的配置参数从 cfg 传入。
            enc = LSTMEncoder(
                vocab_size=len(vocab.itos), emb_dim=cfg.emb_dim, hidden=cfg.hidden,
                num_layers=cfg.layers, bidirectional=cfg.bidirectional,
                dropout=cfg.dropout, pad_idx=vocab.pad
            )
            # 2. 实例化 TextClassifier：用 classifier.py 中的通用分类器外壳，将刚创建的编码器 enc 包装起来。
            #    enc.out_dim 是编码器计算好的输出维度，用于定义分类头部的输入维度。
            model = TextClassifier(enc, enc.out_dim, num_labels=1, dropout=cfg.dropout)
            # 3. 为这个模型类型设置一个标签，方便后续保存文件时命名。
            tag = "lstm"
            # 如果模型类型是 "lstm_lora"
        elif cfg.model_type == "lstm_lora":
            print("Using model: LSTM-LoRA (loralib)")
             # 打印出当前LoRA配置的详细信息，便于调试和记录。
            print(f"LoRA config: r={cfg.lora_r}, alpha={cfg.lora_alpha}, dropout={cfg.lora_dropout}, "
                  f"freeze_base={cfg.freeze_base}, freeze_embed={cfg.freeze_embed}")
            # 1. 实例化 LSTMLoRAEncoder：从 lstm_lora.py 中创建我们手写的、支持LoRA的编码器实例。
            #    除了标准参数外，还传入了 lora_r, lora_alpha 等LoRA专属参数。
            enc = LSTMLoRAEncoder(
                vocab_size=len(vocab.itos), emb_dim=cfg.emb_dim, hidden=cfg.hidden,
                num_layers=cfg.layers,
                bidirectional=cfg.bidirectional,
                dropout=cfg.dropout, pad_idx=vocab.pad,
                lora_r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                freeze_base=cfg.freeze_base, merge_weights=cfg.merge_lora
            )
            # 2. 同样用 TextClassifier 进行包装。
            model = TextClassifier(enc, enc.out_dim, num_labels=1, dropout=cfg.dropout)
            tag = "lstm_lora"
            # --- LoRA专属的可选步骤：加载基座模型权重 ---
            # 3. 检查配置中是否指定了基座模型的路径。
            if cfg.base_model_path:
                print(f"Loading base model weights from: {cfg.base_model_path}")
                try:
                    # a) 使用 torch.load 从硬盘加载标准LSTM模型的权重文件（state_dict）。
                    #    map_location='cpu' 确保权重能被正确加载到CPU上，避免设备不匹配问题。
                    base_model_state = torch.load(cfg.base_model_path, map_location=torch.device('cpu'))
                    # b) 使用 model.load_state_dict() 将加载的权重应用到当前LoRA模型上。
                    #    - strict=False 是这里的关键。它告诉PyTorch：“请尽力加载权重。如果遇到
                    #      名称或尺寸不匹配的权重，请不要报错，直接跳过就好。”
                    #    - 这允许我们成功加载名称匹配的 Embedding层 和 Head层，而忽略名称不匹配的LSTM核心层。
                    model.load_state_dict(base_model_state, strict=False)
                    print("Successfully loaded compatible weights into LoRA model.")
                except Exception as e:
                    print(f"Could not load base model weights. Error: {e}. Proceeding with random weights.")
                    # 如果模型类型不是以上任何一种，则抛出错误。
        else:
            raise ValueError(f"Unknown model_type: {cfg.model_type}")
        # --- LoRA专属的可选步骤：冻结嵌入层 ---
        # 检查是否是LoRA模型，并且配置要求冻结嵌入层。
        if cfg.model_type == "lstm_lora" and cfg.freeze_embed:
            if cfg.base_model_path:
                 print("Embedding layer weights loaded from base model and are now frozen.")
            # a) 找到模型的嵌入层权重 (model.encoder.emb.weight)。
            # b) 将其 requires_grad 属性设置为 False。
            #    - 这会告诉PyTorch的自动求导系统：“在反向传播时，请忽略这个参数，不要为它计算梯度。”
            #    - 从而实现了在训练过程中“冻结”该层，使其参数不被更新。
            model.encoder.emb.weight.requires_grad = False
            print("Freeze embedding: True")
        return model, tag