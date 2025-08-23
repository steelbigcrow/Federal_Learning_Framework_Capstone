# 联邦学习本地串行同步框架（PyTorch + LoRA）

本项目在本机串行、同步地模拟联邦学习（FedAvg），支持以下数据与模型：

- 数据集：IMDB（二分类）、MNIST（10 类）
- 模型：
  - IMDB：RNN、LSTM、Transformer
  - MNIST：MLP、ViT
- 训练：
  - 基于 loralib 的 LoRA 微调（仅训练 LoRA + 分类头）
- 数据分布（label shifting）：
  - IMDB：10 个客户端，其中 5 个仅包含 label=0，5 个仅包含 label=1
  - MNIST：10 个客户端，每个客户端仅包含一种数字类别

## 目录结构

```
Federal Learning Framework/
├─ README.md
├─ requirements.txt
├─ pytest.ini
├─ configs/
│  ├─ arch/
│  │  ├─ imdb_lstm.yaml             # 实际配置文件（不被跟踪）
│  │  ├─ imdb_rnn.yaml              # 实际配置文件（不被跟踪）
│  │  ├─ imdb_transformer.yaml      # 实际配置文件（不被跟踪）
│  │  ├─ mnist_mlp.yaml             # 实际配置文件（不被跟踪）
│  │  ├─ mnist_vit.yaml             # 实际配置文件（不被跟踪）
│  │  ├─ imdb_lstm.yaml.example     # 配置文件示例（被跟踪）
│  │  ├─ imdb_rnn.yaml.example      # 配置文件示例（被跟踪）
│  │  ├─ imdb_transformer.yaml.example # 配置文件示例（被跟踪）
│  │  ├─ mnist_mlp.yaml.example     # 配置文件示例（被跟踪）
│  │  └─ mnist_vit.yaml.example     # 配置文件示例（被跟踪）
│  ├─ federated.yaml               # 实际配置文件（不被跟踪）
│  ├─ federated.yaml.example       # 配置文件示例（被跟踪）
├─ data_cache/                     # 数据缓存目录
├─ src/
│  ├─ datasets/                    # 数据集处理模块
│  │  ├─ imdb.py                   # IMDB数据集处理
│  │  ├─ mnist.py                  # MNIST数据集处理
│  │  ├─ partition.py              # 数据分区策略
│  │  └─ text_utils.py             # 文本处理工具
│  ├─ federated/                   # 联邦学习核心模块
│  │  ├─ aggregator.py             # 模型聚合器
│  │  ├─ client.py                 # 客户端实现
│  │  └─ server.py                 # 服务器实现
│  ├─ models/                      # 模型定义模块
│  │  ├─ imdb_lstm.py              # IMDB LSTM模型
│  │  ├─ imdb_rnn.py               # IMDB RNN模型
│  │  ├─ imdb_transformer.py       # IMDB Transformer模型
│  │  ├─ mnist_mlp.py              # MNIST MLP模型
│  │  ├─ mnist_vit.py              # MNIST ViT模型
│  │  └─ registry.py               # 模型注册器
│  ├─ training/                    # 训练相关模块
│  │  ├─ checkpoints.py            # 检查点管理
│  │  ├─ evaluate.py               # 模型评估
│  │  ├─ logging_utils.py          # 日志工具
│  │  ├─ lora_utils.py             # LoRA工具
│  │  └─ plotting.py               # 绘图工具
│  └─ utils/                       # 通用工具模块
│     ├─ config.py                 # 配置管理
│     ├─ device.py                 # 设备管理
│     ├─ paths.py                  # 路径管理
│     ├─ seed.py                   # 随机种子设置
│     └─ serialization.py          # 序列化工具
├─ scripts/
│  ├─ fed_train.py                 # 联邦训练主脚本
│  ├─ inspect_checkpoint.py        # 检查点检查脚本
└─ outputs/                        # 输出目录
   ├─ checkpoints/                 # 模型检查点
   ├─ logs/                        # 训练日志
   ├─ loras/                       # LoRA权重文件
   ├─ metrics/                     # 评估指标
   └─ plots/                       # 可视化图表
```

- **数据集处理**：通过 `datasets` 库在线获取并缓存至 `./data_cache` 目录
- **模型架构**：支持 IMDB（RNN、LSTM、Transformer）和 MNIST（MLP、ViT）模型
- **联邦学习**：实现 FedAvg 算法，支持标准训练和 LoRA 微调
- **配置管理**：采用双配置模式，分离架构配置和训练配置
- **输出管理**：结构化的输出目录，按功能分类存储各种文件

## 快速开始

1) 安装依赖

```
pip install -r requirements.txt
```

2) 双配置用法

使用两个 YAML 文件：
- **架构配置**（`configs/arch/`）：定义数据集和模型结构参数
- **训练配置**（`configs/federated.yaml`）：定义联邦学习超参数、LoRA设置等

**示例 A：IMDB + Transformer 联邦 LoRA 微调**

```terminal
python scripts/fed_train.py --arch-config configs/arch/imdb_transformer.yaml --train-config configs/federated.yaml --use-lora
```

**示例 B：MNIST + MLP 标准联邦训练**

```terminal
python scripts/fed_train.py --arch-config configs/arch/mnist_mlp.yaml --train-config configs/federated.yaml
```

**示例 C：MNIST + ViT LoRA 联邦微调**

```terminal
python scripts/fed_train.py --arch-config configs/arch/mnist_vit.yaml --train-config configs/federated.yaml --use-lora
```

**配置覆盖**：使用 `--override` 参数临时覆盖配置项：

```bash
--override federated.num_rounds=5 federated.num_clients=5 lora.r=16
```

**数据缓存控制**：
- 默认启用缓存：`--data-cache-dir ./data_cache`
- 禁用缓存：`--no-cache`（直接从 HuggingFace 加载）



## 联邦学习输出

框架支持两种训练模式：

### 1. 标准联邦训练
- **训练方式**：全模型参数训练
- **客户端权重**：`outputs/checkpoints/{dataset}_{model}_{timestamp}/clients/client_{id}/round_{r}.pth`
- **全局权重**：`outputs/checkpoints/{dataset}_{model}_{timestamp}/server/round_{r}.pth`

### 2. LoRA 联邦微调
- **训练方式**：仅训练 LoRA 权重和分类头，基模参数冻结
- **LoRA 权重**：`outputs/loras/{dataset}_{model}_lora_{timestamp}/clients/client_{id}/lora_round_{r}.pth`
- **全局 LoRA**：`outputs/loras/{dataset}_{model}_lora_{timestamp}/server/lora_round_{r}.pth`
- **基模路径**：在 `federated.yaml` 中指定，用于 LoRA 微调的基模加载

### 通用输出文件

**评估指标**：`outputs/metrics/{run_name}/client_{id}/round_{r}_metrics.json`
- 包含训练损失、验证准确率等指标
- 支持全局模型评估指标

**训练日志**：`outputs/logs/{run_name}/client_{id}/round_{r}_training.log`
- 详细的训练过程日志
- 包含每个 epoch 的损失和指标

**可视化图表**：`outputs/plots/{run_name}/client_{id}/client_{id}_metrics.png`
- 自动生成的训练过程图表
- 展示损失和准确率曲线

### 输出路径说明

- `{dataset}`：数据集名称（mnist/imdb）
- `{model}`：模型名称（lstm/rnn/transformer/mlp/vit）
- `{timestamp}`：训练开始时间戳（格式：YYYYMMDD_HHMMSS）
- `{run_name}`：运行名称（来自配置中的 `run_name` 字段）
- `{id}`：客户端 ID（0-9）
- `{r}`：训练轮次（1-`num_rounds`）

## 核心特性

### LoRA 配置详解

在 `configs/federated.yaml` 中配置 LoRA 参数：

```yaml
lora:
  r: 8                    # LoRA 秩
  alpha: 16               # LoRA 缩放因子
  dropout: 0.05           # LoRA dropout 率
  target_modules: ["Linear", "Embedding"]  # 目标模块类型
  train_classifier_head: true  # 是否训练分类头
  base_model_path: "checkpoints/mnist_mlp_20231215_143022/server/round_10.pth"  # 基模路径（必须指定）
```

**重要提醒**：
- **LoRA微调必须指定基模路径**：`base_model_path` 字段是必需的，不能为空
- **路径格式**：可以是相对路径（相对于outputs目录）或绝对路径
- **配置方式**：只允许通过修改配置文件来设置基模路径，不支持命令行覆盖
- **错误处理**：如果未指定基模路径，程序会报错并退出

### 数据分区策略

**IMDB 数据集**：
- 10 个客户端
- 5 个客户端仅包含负面评论（label=0）
- 5 个客户端仅包含正面评论（label=1）
- 模拟非独立同分布（Non-IID）场景

**MNIST 数据集**：
- 10 个客户端
- 每个客户端仅包含一种数字类别（0-9）
- 完全模拟标签偏移场景

### 设备管理

- **自动检测**：使用 `torch.device('cuda' if torch.cuda.is_available() else 'cpu')`

### 依赖环境

主要依赖包：
- `torch>=2.2.0` - PyTorch 深度学习框架
- `loralib>=0.1.2` - LoRA 实现库
- `datasets>=2.19.0` - HuggingFace 数据集库
- `nltk>=3.8.1` - 自然语言处理工具
- 其他工具：`PyYAML`, `numpy`, `tqdm`, `matplotlib` 等

## 配置管理策略

### 配置文件版本控制

为避免不同主机间的配置冲突，采用以下策略：

1. **示例配置文件**（被版本控制）：
   - `configs/federated.yaml.example` - 标准训练配置示例
   - `configs/arch/*.yaml.example` - 架构配置文件示例

2. **实际配置文件**（不被版本控制）：
   - `configs/federated.yaml` - 实际使用的训练配置
   - `configs/arch/*.yaml` - 实际使用的架构配置

### 配置使用流程

1. **首次设置**：复制示例文件为实际配置文件
   ```bash
   # 复制训练配置文件
   cp configs/federated.yaml.example configs/federated.yaml

   # 复制架构配置文件（根据需要选择）
   cp configs/arch/imdb_transformer.yaml.example configs/arch/imdb_transformer.yaml
   cp configs/arch/mnist_mlp.yaml.example configs/arch/mnist_mlp.yaml
   ```
