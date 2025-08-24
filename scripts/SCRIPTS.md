# Scripts 脚本模块文档

## 概述

`scripts` 模块是联邦学习框架的执行脚本集合，包含了训练和模型检查等核心功能的命令行工具。该模块提供了完整的工作流程支持，从模型训练到结果分析。

## 文件结构

```
scripts/
├── fed_train.py                 # 联邦训练主脚本
├── inspect_checkpoint.py        # 检查点检查脚本
├── eval_final.py                # 离线评测：加载全局权重/基模+LoRA，在 datasets 测试集上推理并出图
├── compare_rounds.py            # 多轮对比：遍历 server/round_*.pth 逐轮评测，画“指标-轮次”曲线
└── fed_train_auto.py            # 训练+自动评测：先跑 fed_train，再自动调用 eval_final
└── SCRIPTS.MD          # 本文档
```

## 各脚本详细说明

### 1. 联邦学习训练脚本

#### fed_train.py - 联邦学习训练主程序
**主要功能：**
- 联邦学习框架的主入口程序
- 支持标准联邦学习和LoRA微调的联邦学习
- 自动处理MNIST和IMDB数据集
- 提供完整的训练流程和日志记录

**核心特性：**
- 灵活的配置文件系统，支持架构配置和训练配置分离
- 自动设备检测（CPU/GPU）
- 集成的LoRA微调支持，减少训练参数量
- 客户端本地训练和全局模型聚合
- 完整的训练日志和检查点管理
- 自动数据下载和缓存处理

**使用示例：**
```bash
# LoRA微调模式训练MNIST数据集
python scripts/fed_train.py --arch-config configs/arch/mnist_mlp.yaml --train-config configs/federated.yaml --use-lora

# 标准训练模式训练IMDB数据集
python scripts/fed_train.py --arch-config configs/arch/imdb_lstm.yaml --train-config configs/federated.yaml

# 禁用数据缓存，直接从HuggingFace下载
python scripts/fed_train.py --arch-config configs/arch/mnist_mlp.yaml --train-config configs/federated.yaml --no-cache

# 指定自定义数据缓存目录
python scripts/fed_train.py --arch-config configs/arch/mnist_mlp.yaml --train-config configs/federated.yaml --data-cache-dir ./custom_cache
```

**命令行参数：**
- `--arch-config`: 架构配置文件路径（必需）
- `--train-config`: 训练配置文件路径（必需）
- `--use-lora`: 启用LoRA微调模式
- `--no-cache`: 禁用数据缓存，直接从HuggingFace下载
- `--data-cache-dir`: 指定数据缓存目录（默认: ./data_cache）
- `--override`: 覆盖配置文件中的特定参数（格式: key=value）

### 2. 检查点文件检查工具

#### inspect_checkpoint.py - 模型检查点分析工具
**主要功能：**
- 检查PyTorch模型检查点文件的内容和结构
- 显示模型状态字典中的键值对信息
- 分析模型参数的形状和维度

**核心特性：**
- 快速加载和解析检查点文件
- 显示状态字典的完整结构
- 支持大型模型文件的内存优化加载

**使用示例：**
```bash
# 检查全局模型检查点
python scripts/inspect_checkpoint.py --path outputs/checkpoints/global_model_round_10.pt

# 检查客户端模型检查点
python scripts/inspect_checkpoint.py --path outputs/checkpoints/mnist_mlp_20241201_120000/clients/client_0/round_5.pth

# 检查LoRA权重文件
python scripts/inspect_checkpoint.py --path outputs/loras/mnist_mlp_lora_20241201_120000/server/lora_round_5.pth
```

**命令行参数：**
- `--path`: 检查点文件路径（必需，支持.pth和.pt格式）

### 3. 离线评测工具

#### eval_final.py - 加载全局权重（或基模+LoRA）并在 datasets 测试集评测
**主要功能：**
- 读取非 LoRA全量权重（server/round_*.pth）或LoRA（基模 + lora_round_*.pth）
- 在 Hugging Face datasets 的官方测试集上自动评测：MNIST：混淆矩阵、每类准确率、误分类样例 ;IMDB：ROC 曲线、PR 曲线
- GPU/CPU 自适应（--device cuda/cpu）

**核心特性：**
- 自动根据 --arch-config 推断任务与模型并构建网络
- 兼容性加载：对参数名/形状不完全一致时尽量过滤/扩容后加载，避免因小改动报错
- 支持大型模型文件的内存优化加载

**使用示例：**
```bash
# 非 LoRA：MNIST-MLP
python scripts/eval_final.py \
  --arch-config configs/arch/mnist_mlp.yaml \
  --checkpoint outputs/checkpoints/mnist_mlp_20250823_203138/server/round_2.pth \
  --outdir outputs/viz/mnist_mlp_20250823_203138/round_2 \
  --device cuda

# LoRA：IMDB-RNN（基模 + LoRA 适配器）
python scripts/eval_final.py \
  --arch-config configs/arch/imdb_rnn.yaml \
  --checkpoint outputs/checkpoints/imdb_rnn_20250823_213408/server/round_1.pth \
  --lora-ckpt outputs/loras/imdb_rnn_lora_20250823_221050/server/lora_round_3.pth \
  --outdir outputs/viz/imdb_rnn_lora_20250823_221050/round_3 \
  --device cuda
```

**命令行参数：**
- `--path`: 检查点文件路径（必需，支持.pth和.pt格式）
- `--arch-config`：架构配置路径（必需）

- `--checkpoint`：评测用 checkpoint；非 LoRA=全量权重，LoRA=基模权重（必需）

- `--lora-ckpt`：LoRA 适配器权重（仅 LoRA 场景需要）

- `--outdir`：评测输出目录（必需）

- `--device`：cuda / cpu（默认自动）

- `--mis-limit`：MNIST 误分类展示上限（默认 36）

### 4. 多轮结果对比

#### compare_rounds.py - 遍历 server/round_*.pth 逐轮评测并画曲线
**主要功能：**
- 针对一次训练运行（一个 run 目录），把 server/round_*.pth 逐轮评测： MNIST：Accuracy vs Round ；MDB：ROC-AUC / PR-AUC vs Round

- 导出曲线图与 metrics.json（逐轮指标明细）。

**核心特性：**
- 自动从 run 目录名与 sidecar json 推断模型与任务。
- 支持选择特定轮次（如 --rounds 1,3,5）。
- 与 eval_final.py 同样具备兼容性加载。

**使用示例：**
```bash
# 对比 MNIST 某次训练的所有轮次
python scripts/compare_rounds.py \
  --run-dir outputs/checkpoints/mnist_mlp_20250823_203138 \
  --device cuda

# 只对比 IMDB 的第 1、3 轮
python scripts/compare_rounds.py \
  --run-dir outputs/checkpoints/imdb_rnn_20250823_213408 \
  --rounds 1,3 \
  --device cuda
```

**命令行参数：**
- `--path`: 检查点文件路径（必需，支持.pth和.pt格式）
- `--run-dir`：一次训练的 run 目录（形如 outputs/checkpoints/<name_时间戳> 或其下的 server/ 目录）（必需）

- `--rounds`：逗号分隔的轮次子集（可选；默认所有轮次）

- `--device`：cuda/cpu（默认自动）

- `--out-root`：输出根目录（默认 outputs/viz；最终会写入 <out-root>/<run_name>/compare_rounds/）

### 5. 训练 + 自动评测（一步到位）

#### fed_train_auto.py - 先训练，再自动调用 eval_final.py
**主要功能：**
- 完整封装一次流程：训练完成后自动评测。
- 非 LoRA：自动寻找 outputs/checkpoints/**/server/round_R.pth（R 为配置里的 num_rounds）。
- LoRA：自动寻找 outputs/loras/**/server/lora_round_R.pth，并从 configs/federated.yaml > lora.base_model_path 读取基模。

**核心特性：**
- 参数与 fed_train.py 完全一致（可以直接把命令里的文件名从 fed_train.py 换成 fed_train_auto.py）。
- 自动决定调用 eval_final.py 的参数（包括 LoRA/非 LoRA 分支）。

**使用示例：**
```bash
# 标准训练（MNIST-MLP）2 轮并自动评测
python scripts/fed_train_auto.py \
  --arch-config configs/arch/mnist_mlp.yaml \
  --train-config configs/federated.yaml \
  --override run_name=mnist_mlp_base federated.num_rounds=2 federated.local_epochs=1

# LoRA 训练（IMDB-RNN）3 轮并自动评测
# 注意：在 configs/federated.yaml 中需先设置 lora.base_model_path 指向一份基模 server/round_*.pth
python scripts/fed_train_auto.py \
  --arch-config configs/arch/imdb_rnn.yaml \
  --train-config configs/federated.yaml \
  --use-lora \
  --override run_name=imdb_rnn_lora federated.num_rounds=3 federated.local_epochs=1

```

**命令行参数：**
- `--path`: 检查点文件路径（必需，支持.pth和.pt格式） 
- 与`fed_train.py` 一致：--arch-config、--train-config、--override、--use-lora、--no-cache、--data-cache-dir 等

- 有`--viz-name（可选）`：自定义评测输出目录名（默认使用 run_name）
## 工作流程

### 联邦学习训练完整流程

1. **训练配置阶段**：
   - 准备架构配置文件（configs/arch/）
   - 准备训练配置文件（configs/federated.yaml）
   - LoRA配置集成在federated.yaml中，无需单独文件

2. **模型训练阶段**：
   - 使用 `fed_train.py` 执行联邦学习训练
   - 支持自动数据下载和缓存
   - 监控训练进度和性能指标
   - 自动保存检查点和日志

3. **结果分析阶段**：
   - 使用 `inspect_checkpoint.py` 检查训练结果
   - 分析模型性能和收敛情况

### 脚本依赖关系

```
训练配置 ───────────┬─ 配置文件准备
                     │
联邦学习训练 ───────┼─ fed_train.py 主训练脚本
                     │
结果检查 ───────────┘─ inspect_checkpoint.py 检查训练结果
```

## 配置要求

### 环境依赖
- Python 3.7+
- PyTorch
- 相关深度学习库（torchvision, transformers等）
- 联邦学习框架依赖包

### 文件路径
- 输出目录：`./outputs`（可通过配置修改）
- 数据缓存：`./data_cache`（自动创建和使用）
- 配置文件：`./configs` 目录

## 注意事项

1. **数据缓存**：脚本会自动处理数据下载和缓存，无需手动预处理
2. **GPU使用**：脚本会自动检测可用设备（GPU/CPU）
3. **内存管理**：处理大型模型时注意内存使用，检查点文件可能很大
4. **配置文件**：确保配置文件路径正确，参数设置合理
5. **LoRA配置**：LoRA微调必须在配置文件中指定有效的base_model_path

## 故障排除

- **数据下载失败**：检查网络连接，使用 `--no-cache` 禁用缓存并直接下载
- **CUDA错误**：确保GPU内存充足，尝试减少batch_size
- **配置文件错误**：验证YAML格式，检查必需的配置项
- **LoRA基模路径错误**：确保base_model_path指向有效的检查点文件
- **内存不足**：减少batch_size或使用CPU训练
