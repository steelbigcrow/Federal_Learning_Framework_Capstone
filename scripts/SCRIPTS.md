# Scripts 脚本模块文档

## 概述

`scripts` 模块包含联邦学习框架的命令行执行脚本，提供训练和模型检查功能。

## 文件结构

```
scripts/
├── fed_train.py         # 联邦学习训练主脚本
├── inspect_checkpoint.py # 检查点文件检查工具
└── SCRIPTS.md          # 本文档
```

## 各脚本详细说明

### fed_train.py - 联邦学习训练主脚本

**功能：**
- 联邦学习框架的主入口程序
- 支持标准联邦学习和LoRA微调模式
- 处理MNIST和IMDB数据集
- 提供完整的训练流程和日志记录

**命令行参数：**
- `--arch-config`: 架构配置文件路径（必需）
- `--train-config`: 训练配置文件路径（必需）
- `--use-lora`: 启用LoRA微调模式
- `--no-cache`: 禁用数据缓存
- `--data-cache-dir`: 指定数据缓存目录（默认: ./data_cache）
- `--override`: 覆盖配置文件参数（格式: key=value key2=value2）

**使用示例：**
```bash
# LoRA微调模式
python scripts/fed_train.py --arch-config configs/arch/mnist_mlp.yaml --train-config configs/federated.yaml --use-lora

# 标准训练模式
python scripts/fed_train.py --arch-config configs/arch/imdb_transformer.yaml --train-config configs/federated.yaml

# 禁用数据缓存
python scripts/fed_train.py --arch-config configs/arch/mnist_mlp.yaml --train-config configs/federated.yaml --no-cache
```

### inspect_checkpoint.py - 检查点文件检查工具

**功能：**
- 检查PyTorch模型检查点文件的内容
- 显示状态字典中的键数量
- 显示前50个键值对的张量形状信息

**命令行参数：**
- `--path`: 检查点文件路径（必需，支持.pth和.pt格式）

**使用示例：**
```bash
# 检查检查点文件
python scripts/inspect_checkpoint.py --path outputs/checkpoints/mnist_mlp_20241201_120000/server/round_5.pth
```
## 配置要求

### 环境依赖
- Python 3.7+
- PyTorch
- loralib
- datasets (HuggingFace)
- 其他依赖见requirements.txt

### 配置文件
- 架构配置：`configs/arch/` 目录下的YAML文件
- 训练配置：`configs/federated.yaml`
- LoRA配置：集成在federated.yaml中的lora部分

### 输出路径
- 检查点：`./outputs/checkpoints/`
- LoRA权重：`./outputs/loras/`
- 训练日志：`./outputs/logs/`
- 数据缓存：`./data_cache/`
