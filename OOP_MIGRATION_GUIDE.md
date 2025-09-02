# OOP架构迁移指南

## 概述

本指南说明如何从旧的联邦学习架构迁移到新的OOP架构。新架构提供了更好的模块化、可扩展性和可维护性，同时保持完全的向后兼容性。

## 架构对比

### 旧架构特点
- 过程式编程风格
- 功能分散在多个模块中
- 硬编码的聚合逻辑
- 有限的扩展性

### 新OOP架构特点
- 面向对象设计
- 清晰的分层架构
- 策略模式和工厂模式
- 插件系统支持
- 完整的异常处理

## 迁移策略

### 方案1：渐进式迁移（推荐）

当前实现已经采用了这种方案，新旧架构并存：

```python
# 旧API仍然可用
from src.federated import Server, Client

# 新OOP API也可用
from src.federated import FederatedServer, FederatedClient
from src.factories import ComponentFactory
```

#### 优势
- 零破坏性变更
- 用户可以逐步迁移
- 保持现有脚本继续工作

### 方案2：完全迁移

使用新的入口脚本 `fed_train_oop.py`：

```bash
# 使用工厂模式（默认）
python scripts/fed_train_oop.py --arch-config configs/arch/mnist_mlp.yaml --train-config configs/federated.yaml

# 使用策略模式
python scripts/fed_train_oop.py --use-strategy-pattern --arch-config configs/arch/mnist_mlp.yaml --train-config configs/federated.yaml

# 使用LoRA
python scripts/fed_train_oop.py --use-lora --arch-config configs/arch/mnist_mlp.yaml --train-config configs/federated.yaml

# 使用AdaLoRA
python scripts/fed_train_oop.py --use-adalora --arch-config configs/arch/mnist_mlp.yaml --train-config configs/federated.yaml
```

## 新架构使用方式

### 1. 工厂模式（推荐用于快速设置）

```python
from src.factories.component_factory import ComponentFactory

# 创建工厂
factory = ComponentFactory()

# 一站式创建联邦学习系统
config = {
    'dataset': {
        'name': 'mnist',
        'config': {
            'batch_size': 32,
            'num_clients': 10,
            'partition_method': 'label_shift'
        }
    },
    'model': {
        'constructor': create_model_function,
        'config': model_config
    },
    'server': {
        'type': 'federated_server',
        'config': server_config
    }
}

server = factory.create_federated_system(config)
server.run(num_rounds=10, local_epochs=5)
```

### 2. 策略模式（推荐用于高级定制）

```python
from src.implementations.servers import FederatedServer
from src.strategies.aggregation import FedAvgStrategy, LoRAFedAvgStrategy
from src.strategies.training import StandardTrainingStrategy, LoRATrainingStrategy

# 选择策略
aggregation_strategy = LoRAFedAvgStrategy()
training_strategy = LoRATrainingStrategy(lora_config)

# 创建组件
clients = [create_client(...) for _ in range(num_clients)]
server = FederatedServer(
    model_constructor=model_constructor,
    clients=clients,
    aggregation_strategy=aggregation_strategy,
    ...
)

# 运行时可以切换策略
server.set_aggregation_strategy(FedAvgStrategy())
```

### 3. 组件化创建（最大灵活性）

```python
from src.implementations.servers import FederatedServer
from src.implementations.clients import FederatedClient
from src.implementations.aggregators import FederatedAggregator

# 单独创建每个组件
clients = []
for i in range(num_clients):
    client = FederatedClient(
        client_id=i,
        train_loader=train_loaders[i],
        model_constructor=model_constructor,
        device=device,
        config=optimizer_config
    )
    clients.append(client)

aggregator = FederatedAggregator()
server = FederatedServer(
    model_constructor=model_constructor,
    clients=clients,
    aggregator=aggregator,
    ...
)
```

## 功能映射表

| 旧架构功能 | 新架构实现 | 使用方式 |
|------------|------------|----------|
| 标准联邦学习 | FederatedServer + FedAvgStrategy | 默认模式 |
| LoRA微调 | FederatedServer + LoRA配置 | 自动检测 |
| AdaLoRA微调 | FederatedServer + AdaLoRA配置 | 自动检测 |
| 模型聚合 | Strategy Pattern | 可插拔策略 |
| 客户端管理 | FederatedServer.clients | 统一管理 |
| 检查点保存 | PathManager | 路径抽象 |
| 配置管理 | Config Validation | 自动验证 |

## 新增功能

### 1. 策略系统
- 聚合策略可插拔
- 训练策略可定制
- 策略性能监控

### 2. 工厂系统
- 统一组件创建
- 配置验证
- 兼容性检查

### 3. 插件系统
- 功能扩展插件
- 指标收集插件
- 可视化插件

### 4. 异常处理
- 分层异常体系
- 详细错误信息
- 自动恢复机制

## 迁移步骤

### 步骤1：评估现有代码
```python
# 检查当前使用的API
from src.federated import Server, Client  # 旧API
```

### 步骤2：选择迁移策略
- 渐进式：保持现有代码，逐步使用新功能
- 完全迁移：切换到新的入口脚本

### 步骤3：更新配置（如需要）
新架构完全兼容旧配置，无需修改。

### 步骤4：测试
```bash
# 运行测试确保功能正常
python scripts/fed_train.py --arch-config configs/arch/mnist_mlp.yaml --train-config configs/federated.yaml --override federated.num_rounds=1

# 或使用新脚本
python scripts/fed_train_oop.py --arch-config configs/arch/mnist_mlp.yaml --train-config configs/federated.yaml --override federated.num_rounds=1
```

## 最佳实践

### 1. 新项目
直接使用新OOP架构：
```python
# 推荐使用工厂模式快速开始
from src.factories.component_factory import ComponentFactory
factory = ComponentFactory()
server = factory.create_federated_system(config)
```

### 2. 现有项目
- 保持现有入口脚本不变
- 逐步在代码中使用新架构的特性
- 例如：使用策略模式添加新的聚合算法

### 3. 扩展功能
```python
# 添加新的聚合策略
class CustomAggregationStrategy(AggregationStrategyInterface):
    def aggregate(self, models, weights):
        # 实现自定义聚合逻辑
        pass

# 注册并使用
server.set_aggregation_strategy(CustomAggregationStrategy())
```

## 常见问题

### Q: 是否必须迁移？
A: 不是必须的。旧API仍然完全可用，新架构保持向后兼容。

### Q: 迁移会影响性能吗？
A: 不会。新架构使用相同的底层实现，只是提供了更好的组织结构。

### Q: 如何处理自定义的聚合算法？
A: 新架构通过策略模式支持自定义算法，更容易集成。

### Q: 配置文件需要修改吗？
A: 不需要。新架构完全兼容现有的配置文件格式。

## 总结

新OOP架构提供了：
- ✅ 完全的功能对等
- ✅ 向后兼容性
- ✅ 更好的可扩展性
- ✅ 更清晰的代码组织
- ✅ 更强的可测试性

迁移是可选的，但推荐新项目使用新架构以获得更好的开发体验。