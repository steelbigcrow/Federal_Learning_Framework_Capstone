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
│  │  ├─ imdb_lstm.yaml              # 实际配置（不被跟踪）
│  │  ├─ imdb_rnn.yaml               # 实际配置（不被跟踪）
│  │  ├─ imdb_transformer.yaml       # 实际配置（不被跟踪）
│  │  ├─ mnist_mlp.yaml              # 实际配置（不被跟踪）
│  │  ├─ mnist_vit.yaml              # 实际配置（不被跟踪）
│  │  ├─ imdb_lstm.yaml.example      # 示例（被跟踪）
│  │  ├─ imdb_rnn.yaml.example       # 示例（被跟踪）
│  │  ├─ imdb_transformer.yaml.example# 示例（被跟踪）
│  │  ├─ mnist_mlp.yaml.example      # 示例（被跟踪）
│  │  └─ mnist_vit.yaml.example      # 示例（被跟踪）
│  ├─ federated.yaml                 # 实际配置（不被跟踪）
│  └─ federated.yaml.example         # 示例（被跟踪）
├─ data_cache/                       # 数据缓存目录
├─ src/
│  ├─ datasets/                      # 数据集处理
│  ├─ federated/                     # 联邦核心（Client/Server/Aggregator）
│  ├─ models/                        # 模型定义 & 注册器
│  ├─ training/                      # 训练 & LoRA 工具
│  └─ utils/                         # 通用工具
├─ scripts/
│  ├─ fed_train.py                   # 联邦训练主脚本
│  ├─ inspect_checkpoint.py          # 检查点查看工具
│  ├─ eval_final.py                  # 离线评测：加载全局权重/基模+LoRA，在 datasets 测试集上推理并出图
│  ├─ compare_rounds.py              # 多轮对比：遍历 server/round_*.pth 逐轮评测，画“指标-轮次”曲线（非 LoRA）
│  └─ fed_train_auto.py              # 训练+自动评测：先跑 fed_train，再自动调用 eval_final
└─ outputs/
   ├─ checkpoints/                   # 标准训练权重（server/round_*.pth）
   ├─ loras/                         # LoRA 权重（server/lora_round_*.pth）
   ├─ logs/                          # 训练日志
   ├─ metrics/                       # 指标
   ├─ plots/                         # 训练曲线
   └─ viz/                           # 评测图表（eval_final / compare_rounds 生成）

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

## 联邦学习测评
### 1. 手动测评联邦学习结果
### 1)非 LoRA（直接用全量 checkpoint）
python scripts/eval_final.py --arch-config configs/arch/mnist_mlp.yaml --checkpoint outputs/checkpoints/mnist_mlp_20250823_203138/server/round_2.pth --outdir outputs/viz/mnist_mlp_20250823_203138/round_2 --device cuda
MNIST：会生成 mnist_confusion_matrix.png、mnist_per_class_acc.png、mnist_misclassified.png 等
IMDB：会生成 imdb_roc.png、imdb_pr.png 同时会输出 metrics.json

### 2) LoRA（传入“基模 + LoRA 适配器”）
python scripts/eval_final.py --arch-config configs/arch/imdb_rnn.yaml --checkpoint outputs/checkpoints/imdb_rnn_20250823_213408/server/round_1.pth --lora-ckpt outputs/loras/imdb_rnn_lora_20250823_221050/server/lora_round_3.pth --outdir outputs/viz/imdb_rnn_lora_20250823_221050/round_3 --device cuda
eval_final.py 会自动根据 --arch-config 推断任务与模型；若权重键名/形状略有差异，会做兼容性加载（过滤不匹配、必要时扩容词嵌入），避免报错中断。

### 2. 多轮对比（非 LoRA）
把同一 run 下不同轮次的全局模型拿来统一评测并出“指标-轮次”曲线：
python scripts/compare_rounds.py --run-dir outputs/checkpoints/mnist_mlp_20250823_203138 --device cuda
# 或只对指定轮次作图：
python scripts/compare_rounds.py --run-dir outputs/checkpoints/mnist_mlp_20250823_203138 --rounds 1,2,3 --device cuda


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

**离线评测可视化**：
`MNIST：outputs/viz/{run}/round_{r}/mnist_confusion_matrix.png、mnist_per_class_acc.png、mnist_misclassified.png`
`IMDB：outputs/viz/{run}/round_{r}/imdb_roc.png、imdb_pr.png`
`汇总指标：outputs/viz/{run}/round_{r}/metrics.json`

**多轮对比图表**：
`MNIST：outputs/viz/{run}/compare_rounds/mnist_acc_vs_round.png + metrics.json`
`IMDB：outputs/viz/{run}/compare_rounds/imdb_auc_vs_round.png + metrics.json`

### 输出路径说明

- `{dataset}`：数据集名称（mnist/imdb）
- `{model}`：模型名称（lstm/rnn/transformer/mlp/vit）
- `{timestamp}`：训练开始时间戳（格式：YYYYMMDD_HHMMSS）
- `{run_name}`：运行名称（来自配置中的 `run_name` 字段）
- `{run}`：一次运行的目录名，通常为 {dataset}_{model}_{timestamp}；LoRA 运行常见为 {dataset}_{model}_lora_{timestamp}
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
