# Scripts 脚本模块文档

## 概述

`scripts` 模块包含联邦学习框架的命令行执行脚本，提供训练和模型检查功能。

## 文件结构

```
scripts/
├── fed_train.py                 # 联邦训练主脚本（支持自动评估）
├── inspect_checkpoint.py        # 检查点检查脚本
├── compare_rounds.py            # 多轮对比：遍历 server/round_*.pth 逐轮评测，画"指标-轮次"曲线
└── SCRIPTS.MD          # 本文档
```

**注意：** `eval_final.py` 脚本已被移除，其功能已完全整合到 `src.evaluation` 模块中。如需单独评估检查点，可通过编程方式使用 `src.evaluation.evaluate_checkpoint()` 函数。

## 各脚本详细说明

### fed_train.py - 联邦学习训练主脚本

**功能：**
- 联邦学习框架的主入口程序
- 支持标准联邦学习和LoRA微调模式
- 处理MNIST和IMDB数据集
- 提供完整的训练流程和日志记录
- **集成自动评估功能**（推荐使用 `--auto-eval`）

**命令行参数：**
- `--arch-config`: 架构配置文件路径（必需）
- `--train-config`: 训练配置文件路径（必需）
- `--use-lora`: 启用LoRA微调模式
- `--no-cache`: 禁用数据缓存
- `--data-cache-dir`: 指定数据缓存目录（默认: ./data_cache）
- `--auto-eval`: 启用训练后自动评估（**推荐使用**）
- `--override`: 覆盖配置文件参数（格式: key=value key2=value2）

**使用示例：**
```bash
# 标准训练模式（带自动评估）
python scripts/fed_train.py --arch-config configs/arch/mnist_mlp.yaml --train-config configs/federated.yaml --auto-eval

# LoRA微调模式（带自动评估）  
python scripts/fed_train.py --arch-config configs/arch/mnist_mlp.yaml --train-config configs/federated.yaml --use-lora --auto-eval

# 标准训练模式（不带评估）
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

## 评估功能

### 集成评估（推荐）

现在所有评估功能都已整合到 `fed_train.py` 中，使用 `--auto-eval` 参数即可实现训练后自动评测：

```bash
# 标准训练（MNIST-MLP）2 轮并自动评测
python scripts/fed_train.py --arch-config configs/arch/mnist_mlp.yaml --train-config configs/federated.yaml --auto-eval

# LoRA 训练（IMDB-RNN）3 轮并自动评测
python scripts/fed_train.py --arch-config configs/arch/imdb_rnn.yaml --train-config configs/federated.yaml --use-lora --auto-eval
```

### 编程式评估

如需在代码中进行评估，可使用 `src.evaluation` 模块提供的便捷函数：

```python
# 评估单个检查点
from src.evaluation import evaluate_checkpoint

results = evaluate_checkpoint(
    arch_config_path="configs/arch/mnist_mlp.yaml",
    checkpoint_path="outputs/checkpoints/mnist_mlp_20250823_203138/server/round_2.pth",
    output_dir="outputs/viz/mnist_mlp_evaluation"
)

# 训练后自动评估
from src.evaluation import auto_evaluate_training

success = auto_evaluate_training(
    arch_config_path="configs/arch/mnist_mlp.yaml", 
    train_config_path="configs/federated.yaml",
    use_lora=False,
    device="cuda"
)
```
## 工作流程

### 联邦学习训练完整流程

1. **训练配置阶段**：
   - 准备架构配置文件（configs/arch/）
   - 准备训练配置文件（configs/federated.yaml）
   - LoRA配置集成在federated.yaml中，无需单独文件

2. **模型训练阶段**：
   - 使用 `fed_train.py --auto-eval` 执行联邦学习训练和自动评估（**推荐**）
   - 或使用 `fed_train.py` 执行训练后手动评估
   - 支持自动数据下载和缓存
   - 监控训练进度和性能指标
   - 自动保存检查点和日志

3. **结果分析阶段**：
   - 自动评估生成可视化结果（使用 --auto-eval）
   - 使用 `inspect_checkpoint.py` 检查训练结果
   - 使用 `compare_rounds.py` 进行多轮对比分析

### 脚本依赖关系

所有评估功能现已整合到 `src.evaluation` 模块中，通过以下方式访问：
- **集成评估**: `fed_train.py --auto-eval`（推荐）
- **编程评估**: `src.evaluation.evaluate_checkpoint()` 和 `src.evaluation.auto_evaluate_training()`
- **多轮对比**: `compare_rounds.py`

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
- 评估结果：`./outputs/plots/`
- 数据缓存：`./data_cache/`
