# Scripts 脚本模块文档

## 概述

`scripts` 模块是联邦学习框架的执行脚本集合，包含了训练和模型检查等核心功能的命令行工具。该模块提供了完整的工作流程支持，从模型训练到结果分析。

## 文件结构

```
scripts/
├── fed_train.py         # 联邦学习训练主脚本
├── inspect_checkpoint.py # 检查点文件检查工具
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
