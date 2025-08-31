# 联邦学习本地串行同步框架（PyTorch + LoRA + AdaLoRA）

PyTorch 实现的本地串行同步联邦学习框架，支持标准联邦学习、LoRA 微调和 AdaLoRA 微调。

## 特性

- **数据集**：MNIST（10类分类）、IMDB（情感二分类）
- **模型**：
  - MNIST：MLP、Vision Transformer
  - IMDB：RNN、LSTM、Transformer
- **训练模式**：
  - 标准联邦训练（全参数训练）
  - LoRA 微调（仅训练 LoRA + 分类头）
  - AdaLoRA 微调（动态秩分配的自适应微调）
- **数据分布**：强制非独立同分布 Non-IID（10个客户端，与数据集类数匹配）
- **优化导入**：所有模块支持简化的导入模式，减少约54%的导入语句，同时保持完全向后兼容性

## 目录结构

```
federal-learning-framework/
├── configs/
│   ├── arch/
│   │   ├── mnist_mlp.yaml           # ❌ 非跟踪：实际使用的架构配置
│   │   ├── mnist_vit.yaml           # ❌ 非跟踪：实际使用的架构配置
│   │   ├── imdb_rnn.yaml            # ❌ 非跟踪：实际使用的架构配置
│   │   ├── imdb_lstm.yaml           # ❌ 非跟踪：实际使用的架构配置
│   │   ├── imdb_transformer.yaml    # ❌ 非跟踪：实际使用的架构配置
│   │   ├── *.yaml.example           # ✅ 跟踪：版本控制的示例文件
│   ├── federated.yaml               # ❌ 非跟踪：实际使用的训练配置
│   └── federated.yaml.example       # ✅ 跟踪：联邦学习配置示例
├── data_cache/                      # ❌ 非跟踪：数据集缓存（自动生成）
├── outputs/                         # ❌ 非跟踪：训练输出目录
├── scripts/
│   ├── fed_train.py                 # ✅ 跟踪：主训练脚本
│   ├── compare_rounds.py            # ✅ 跟踪：多轮对比分析
│   └── inspect_checkpoint.py        # ✅ 跟踪：检查点检查工具
├── src/                             # ✅ 跟踪：核心源代码
│   ├── datasets/                    # 数据集处理模块
│   ├── federated/                   # 联邦学习核心
│   ├── models/                      # 模型定义与注册器
│   ├── training/                    # 训练工具与LoRA支持
│   ├── evaluation/                  # 综合评估模块
│   └── utils/                       # 工具函数
├── requirements.txt                 # ✅ 跟踪：依赖包列表
├── pytest.ini                       # ✅ 跟踪：测试配置
└── README.md                        # ✅ 跟踪：项目文档
```

**版本控制说明**：
- ✅ 跟踪文件：包含在 Git 版本控制中
- ❌ 非跟踪文件：需要在首次使用时创建，不应提交到仓库

## 快速开始

### 1. 环境安装

```bash
pip install -r requirements.txt
```

### 2. 配置设置

首次使用复制配置文件：
```bash
# 复制训练配置文件
cp configs/federated.yaml.example configs/federated.yaml

# 复制需要的架构配置文件
cp configs/arch/imdb_transformer.yaml.example configs/arch/imdb_transformer.yaml
cp configs/arch/mnist_mlp.yaml.example configs/arch/mnist_mlp.yaml
```

### 3. 核心训练命令

**标准联邦训练：**
```bash
# MNIST MLP 标准训练（推荐）
python scripts/fed_train.py --arch-config configs/arch/mnist_mlp.yaml --train-config configs/federated.yaml --auto-eval

# IMDB Transformer 标准训练
python scripts/fed_train.py --arch-config configs/arch/imdb_transformer.yaml --train-config configs/federated.yaml --auto-eval
```

**LoRA 微调训练：**
```bash
# MNIST MLP LoRA 微调
python scripts/fed_train.py --arch-config configs/arch/mnist_mlp.yaml --train-config configs/federated.yaml --use-lora --auto-eval

# IMDB RNN LoRA 微调
python scripts/fed_train.py --arch-config configs/arch/imdb_rnn.yaml --train-config configs/federated.yaml --use-lora --auto-eval
```

**AdaLoRA 微调训练：**
```bash
# MNIST MLP AdaLoRA 微调
python scripts/fed_train.py --arch-config configs/arch/mnist_mlp.yaml --train-config configs/federated.yaml --use-adalora --auto-eval

# IMDB Transformer AdaLoRA 微调
python scripts/fed_train.py --arch-config configs/arch/imdb_transformer.yaml --train-config configs/federated.yaml --use-adalora --auto-eval
```

### 4. 分析工具

```bash
# 多轮对比分析
python scripts/compare_rounds.py --run-dir outputs/models/mnist_mlp_timestamp --device cuda

# 检查模型权重
python scripts/inspect_checkpoint.py --path outputs/models/mnist_mlp_timestamp/weights/server/round_1.pth
```

## 双配置系统

框架采用双配置文件体系：

### 架构配置（`configs/arch/`）
- 定义数据集和模型结构参数
- 每个模型有专用配置文件
- 支持 `.example` 文件版本控制

### 训练配置（`configs/federated.yaml`）
- 定义联邦学习超参数
- LoRA 设置和管理
- 路径管理和日志设置

## 输出结构

训练输出按时间戳组织，统一结构便于管理：

```
outputs/
├── models/        # 标准模型权重
│   └── {数据集}_{模型架构}_{时间戳}/
│       ├── weights/
│       │   ├── clients/          # 每个客户端的模型权重
│       │   │   └── client_{id}/  # 单独客户端文件夹
│       │   └── server/          # 全局模型权重
│       ├── logs/
│       │   ├── clients/          # 每个客户端的训练日志
│       │   │   └── client_{id}/  # 单独客户端文件夹
│       │   └── server/          # 服务器训练日志
│       ├── metrics/
│       │   ├── clients/          # 每个客户端的训练指标
│       │   │   └── client_{id}/  # 单独客户端文件夹
│       │   └── server/          # 服务器训练指标
│       └── plots/
│           ├── clients/          # 每个客户端的训练图表
│           │   └── client_{id}/  # 单独客户端文件夹
│           └── server/          # 服务器训练图表
└── loras/         # LoRA 适配器权重
    └── {数据集}_{模型架构}_lora_{时间戳}/
        ├── weights/
        │   ├── clients/          # 每个客户端的LoRA权重
        │   │   └── client_{id}/  # 单独客户端文件夹
        │   └── server/          # 全局LoRA权重
        ├── logs/
        │   ├── clients/          # 每个客户端的训练日志
        │   │   └── client_{id}/  # 单独客户端文件夹
        │   └── server/          # 服务器训练日志
        ├── metrics/
        │   ├── clients/          # 每个客户端的训练指标
        │   │   └── client_{id}/  # 单独客户端文件夹
        │   └── server/          # 服务器训练指标
        └── plots/
            ├── clients/          # 每个客户端的训练图表
            │   └── client_{id}/  # 单独客户端文件夹
            └── server/          # 服务器训练图表
└── adaloras/      # AdaLoRA 适配器权重
    └── {数据集}_{模型架构}_adalora_{时间戳}/
        ├── weights/
        │   ├── clients/          # 每个客户端的AdaLoRA权重
        │   │   └── client_{id}/  # 单独客户端文件夹
        │   └── server/          # 全局AdaLoRA权重
        ├── logs/
        │   ├── clients/          # 每个客户端的训练日志
        │   │   └── client_{id}/  # 单独客户端文件夹
        │   └── server/          # 服务器训练日志
        ├── metrics/
        │   ├── clients/          # 每个客户端的训练指标
        │   │   └── client_{id}/  # 单独客户端文件夹
        │   └── server/          # 服务器训练指标
        └── plots/
            ├── clients/          # 每个客户端的训练图表
            │   └── client_{id}/  # 单独客户端文件夹
            └── server/          # 服务器训练图表
```

**结构说明**：
- 每次运行在 `models/`（标准训练）、`loras/`（LoRA训练）或 `adaloras/`（AdaLoRA训练）下创建时间戳文件夹
- 每个运行文件夹包含统一的四个子目录：`weights/`、`logs/`、`metrics/`、`plots/`
- 每个子目录都包含 `clients/` 和 `server/` 两个子目录
- `clients/` 目录下根据客户端数量创建对应数量的 `client_{id}/` 文件夹
- `server/` 目录下不再有子文件夹，直接存储服务器相关文件

**评估可视化已集成**：评估图表现在通过 `fed_train.py --auto-eval` 自动生成到对应运行文件夹的 `plots/` 目录中。

## LoRA 微调特性

- 使用 `loralib` 将 Linear/Embedding 层替换为 LoRA 版本
- 仅训练 LoRA 参数（`lora_A`, `lora_B`）+ 分类头
- 基模型参数在微调期间冻结
- 需在 `federated.yaml` 中设置 `lora.base_model_path`

## AdaLoRA 微调特性

AdaLoRA (Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning) 提供动态秩分配的自适应微调：

- **SVD-based 适配**：使用奇异值分解实现自适应秩分配
- **动态预算管理**：根据参数重要性自动调整各层的秩预算
- **正交正则化**：通过正交约束确保适配器的稳定性
- **预热策略**：支持初始预热和最终微调两阶段训练
- **配置灵活性**：可设置初始秩、目标秩、预热步数等参数
- **独立运行**：与 LoRA 独立，通过 `use_adalora: true` 启用
- **输出目录**：AdaLoRA 权重存储在 `outputs/adaloras/` 目录中
- **基模型要求**：需在 `federated.yaml` 中设置 `adalora.base_model_path`

## 重要实现说明

### 客户端数量要求
- **固定 10 个客户端**：与数据集类数匹配的 Non-IID 数据分布要求
- MNIST：每个客户端获得一种数字类别（0-9）
- IMDB：5个客户端获得负面评论，5个客户端获得正面评论

### 数据分区策略
- **MNIST**：标签偏移完全非IID，模拟最严格的异构性
- **IMDB**：平衡正负分布的定制Non-IID场景

### 参数覆盖
```bash
--override federated.num_rounds=5 federated.num_clients=10 lora.r=8
```

**配置管理最佳实践**：
- 使用 `.example` 文件维护配置模板
- `.yaml.example` 文件受版本控制
- 实际 `.yaml` 文件不进入版本控制，避免环境冲突

### 自动化评测
- `fed_train.py --auto-eval`：集成的训练自动评估工具（推荐使用）
- 自动生成训练曲线、性能指标和对比分析
- 支持标准模型、LoRA 模型和 AdaLoRA 模型的一致性评估

框架专注于非IID联邦学习场景，具有自动设备检测、内存优化加载和CUDA优先支持。