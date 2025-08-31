# 联邦学习框架 OOP 重构指导文档

## 📋 重构概览

**目标**: 将现有的联邦学习框架从过程式编程重构为面向对象设计，提升代码的可扩展性、可维护性和可测试性。

**重构原则**: 遵循SOLID原则，采用设计模式，实现高内聚低耦合的架构设计。

---

## 🏗️ 当前架构分析

### 现有优势
- ✅ 功能完整：支持多种模型(MLP, ViT, RNN, LSTM, Transformer)
- ✅ 多数据集支持：MNIST(图像分类)、IMDB(文本情感分析)
- ✅ 多训练模式：标准联邦学习、LoRA微调、AdaLoRA微调
- ✅ 非IID数据分布：10客户端固定配置
- ✅ 完整输出管理：权重、日志、指标、可视化

### 现存问题
- ❌ 代码耦合度高：组件间依赖关系复杂
- ❌ 扩展性有限：添加新组件需要修改多个文件
- ❌ 缺乏抽象层：没有统一的接口定义
- ❌ 配置管理分散：配置逻辑分布在多个模块
- ❌ 错误处理不统一：异常处理策略不一致

---

## 🎯 重构目标架构

### 核心设计原则
1. **单一职责原则 (SRP)**: 每个类只负责一个功能
2. **开闭原则 (OCP)**: 对扩展开放，对修改关闭
3. **里氏替换原则 (LSP)**: 子类可以替换父类
4. **接口隔离原则 (ISP)**: 客户端不应依赖不需要的接口
5. **依赖倒置原则 (DIP)**: 依赖抽象而非具体实现

### 设计模式应用
- **抽象工厂模式**: 统一组件创建
- **策略模式**: 聚合算法和训练策略
- **观察者模式**: 事件驱动的监控系统
- **装饰者模式**: LoRA/AdaLoRA功能扩展
- **建造者模式**: 复杂配置对象构建

---

## 📁 目标目录结构

```
src/
├── core/                   # 🏗️ 核心抽象层
│   ├── __init__.py
│   ├── base/              # 抽象基类定义
│   │   ├── component.py   # FederatedComponent基类
│   │   ├── client.py      # AbstractClient基类
│   │   ├── server.py      # AbstractServer基类
│   │   └── aggregator.py  # AbstractAggregator基类
│   ├── interfaces/        # 接口定义
│   │   ├── factory.py     # 工厂接口
│   │   ├── strategy.py    # 策略接口
│   │   └── plugin.py      # 插件接口
│   └── exceptions/        # 自定义异常
│       └── exceptions.py
├── strategies/            # 🎯 策略模式实现
│   ├── __init__.py
│   ├── aggregation/       # 聚合策略
│   │   ├── fedavg.py     # 标准FedAvg
│   │   ├── lora_fedavg.py # LoRA联邦平均
│   │   └── adalora_fedavg.py # AdaLoRA联邦平均
│   └── training/          # 训练策略
│       ├── standard.py    # 标准训练
│       ├── lora.py       # LoRA训练
│       └── adalora.py    # AdaLoRA训练
├── factories/             # 🏭 工厂模式
│   ├── __init__.py
│   ├── component_factory.py  # 主工厂
│   ├── model_factory.py     # 模型工厂
│   ├── dataset_factory.py   # 数据集工厂
│   ├── client_factory.py    # 客户端工厂
│   └── server_factory.py    # 服务器工厂
├── config/                # ⚙️ 配置管理系统
│   ├── __init__.py
│   ├── manager.py         # ConfigManager
│   ├── validator.py       # 配置验证器
│   ├── schema.py         # 配置模式定义
│   └── loader.py         # 配置加载器
├── monitoring/            # 📊 监控与日志系统
│   ├── __init__.py
│   ├── metrics/          # 指标收集
│   │   ├── collector.py  # MetricsCollector
│   │   └── aggregator.py # 指标聚合
│   ├── logging/          # 日志系统
│   │   ├── logger.py     # 统一日志器
│   │   └── formatters.py # 日志格式化
│   ├── visualization/    # 可视化
│   │   ├── plotter.py   # 图表生成
│   │   └── dashboard.py # 仪表板
│   └── events/           # 事件系统
│       ├── publisher.py  # 事件发布器
│       └── subscriber.py # 事件订阅器
├── plugins/               # 🔌 插件扩展系统
│   ├── __init__.py
│   ├── manager.py        # PluginManager
│   ├── base.py          # 插件基类
│   ├── lora/            # LoRA插件
│   │   ├── __init__.py
│   │   ├── plugin.py
│   │   └── utils.py
│   └── adalora/         # AdaLoRA插件
│       ├── __init__.py
│       ├── plugin.py
│       └── utils.py
├── implementations/       # 🔧 具体实现
│   ├── __init__.py
│   ├── clients/          # 客户端实现
│   │   └── federated_client.py
│   ├── servers/          # 服务器实现
│   │   └── federated_server.py
│   └── aggregators/      # 聚合器实现
├── models/               # 🤖 模型定义 (保持现有结构)
├── datasets/             # 📊 数据集处理 (保持现有结构)
└── utils/                # 🛠️ 工具函数 (重构优化)
```

---

## 📈 重构实施计划

### 🏁 阶段 0: 准备阶段 (已完成)
- [x] 创建重构指导文档
- [x] 分析现有架构问题
- [x] 设计目标架构

### 🏗️ 阶段 1: 核心抽象层构建 (进行中)
**目标**: 建立框架的抽象基础
- [ ] 创建核心抽象基类
  - [ ] `FederatedComponent` - 联邦学习组件基类
  - [ ] `AbstractClient` - 客户端抽象类
  - [ ] `AbstractServer` - 服务器抽象类  
  - [ ] `AbstractAggregator` - 聚合器抽象类
- [ ] 定义核心接口
  - [ ] `FactoryInterface` - 工厂接口
  - [ ] `StrategyInterface` - 策略接口
  - [ ] `PluginInterface` - 插件接口
- [ ] 创建异常体系
  - [ ] `FederatedLearningException` - 基础异常
  - [ ] 具体异常类定义

**验收标准**: 
- 抽象类和接口定义完整
- 通过基础单元测试
- 文档完整

### 🎯 阶段 2: 策略模式重构
**目标**: 将聚合算法和训练逻辑重构为策略模式
- [ ] 聚合策略实现
  - [ ] `FedAvgStrategy` - 标准联邦平均
  - [ ] `LoRAFedAvgStrategy` - LoRA联邦平均
  - [ ] `AdaLoRAFedAvgStrategy` - AdaLoRA联邦平均
- [ ] 训练策略实现
  - [ ] `StandardTrainingStrategy` - 标准训练
  - [ ] `LoRATrainingStrategy` - LoRA训练  
  - [ ] `AdaLoRATrainingStrategy` - AdaLoRA训练
- [ ] 策略管理器
  - [ ] `StrategyRegistry` - 策略注册器
  - [ ] `StrategyFactory` - 策略工厂

**验收标准**:
- 策略可以动态切换
- 现有功能保持不变
- 新增策略容易扩展

### 🏭 阶段 3: 工厂模式实现
**目标**: 统一组件创建和管理
- [ ] 组件工厂实现
  - [ ] `ComponentFactory` - 主工厂
  - [ ] `ModelFactory` - 模型工厂
  - [ ] `DatasetFactory` - 数据集工厂
  - [ ] `ClientFactory` - 客户端工厂
  - [ ] `ServerFactory` - 服务器工厂
- [ ] 工厂注册系统
  - [ ] `FactoryRegistry` - 工厂注册器
  - [ ] 动态工厂发现和注册

**验收标准**:
- 组件创建统一管理
- 支持配置驱动的组件创建
- 易于添加新的组件类型

### ⚙️ 阶段 4: 配置管理系统
**目标**: 统一配置管理和验证
- [ ] 配置管理核心
  - [ ] `ConfigManager` - 配置管理器
  - [ ] `ConfigValidator` - 配置验证器
  - [ ] `ConfigSchema` - 配置模式定义
- [ ] 配置加载和处理
  - [ ] `ConfigLoader` - 多格式配置加载
  - [ ] `ConfigProcessor` - 配置预处理
  - [ ] `ConfigMerger` - 配置合并器

**验收标准**:
- 配置验证完整准确
- 支持多种配置格式
- 配置错误提示清晰

### 📊 阶段 5: 监控日志系统
**目标**: 建立全面的监控和日志系统
- [ ] 指标收集系统
  - [ ] `MetricsCollector` - 指标收集器
  - [ ] `MetricsAggregator` - 指标聚合器
  - [ ] `MetricsExporter` - 指标导出器
- [ ] 日志管理系统
  - [ ] `UnifiedLogger` - 统一日志器
  - [ ] `LogFormatter` - 日志格式化器
  - [ ] `LogHandler` - 日志处理器
- [ ] 可视化系统
  - [ ] `Visualizer` - 可视化组件
  - [ ] `Dashboard` - 实时仪表板
- [ ] 事件系统
  - [ ] `EventPublisher` - 事件发布器
  - [ ] `EventSubscriber` - 事件订阅器

**验收标准**:
- 监控数据实时收集
- 日志格式统一标准
- 可视化界面友好

### 🔌 阶段 6: 插件扩展系统
**目标**: 建立灵活的插件扩展架构
- [ ] 插件管理核心
  - [ ] `PluginManager` - 插件管理器
  - [ ] `PluginLoader` - 插件加载器
  - [ ] `PluginRegistry` - 插件注册器
- [ ] 现有扩展插件化
  - [ ] `LoRAPlugin` - LoRA功能插件
  - [ ] `AdaLoRAPlugin` - AdaLoRA功能插件
- [ ] 插件接口标准化
  - [ ] `PluginInterface` - 插件标准接口
  - [ ] `PluginLifecycle` - 插件生命周期管理

**验收标准**:
- 插件可以动态加载/卸载
- 插件间隔离良好
- 插件开发文档完整

### 🔧 阶段 7: 具体实现重构
**目标**: 重构现有的客户端和服务器实现
- [ ] 客户端实现重构
  - [ ] `FederatedClient` - 基于新架构的客户端
  - [ ] 客户端工厂集成
  - [ ] 客户端策略集成
- [ ] 服务器实现重构
  - [ ] `FederatedServer` - 基于新架构的服务器
  - [ ] 服务器工厂集成  
  - [ ] 服务器策略集成
- [ ] 聚合器实现重构
  - [ ] 基于策略模式的聚合器重写

**验收标准**:
- 功能完全兼容现有版本
- 性能不低于现有实现
- 代码结构清晰易维护

### 📦 阶段 8: 迁移和集成
**目标**: 完成新旧架构的平滑迁移
- [ ] API兼容层
  - [ ] 保持现有脚本接口不变
  - [ ] 渐进式迁移支持
- [ ] 测试完善
  - [ ] 单元测试覆盖
  - [ ] 集成测试验证
  - [ ] 性能基准测试
- [ ] 文档更新
  - [ ] API文档更新
  - [ ] 使用指南更新
  - [ ] 开发者文档完善

**验收标准**:
- 所有现有功能正常运行
- 测试覆盖率≥90%
- 文档完整准确

---

## 📊 重构进度跟踪

### 总体进度
- 🏁 **阶段 0**: ✅ 已完成 (2024-12-31)
- 🏗️ **阶段 1**: 🟡 进行中 (0/10)
- 🎯 **阶段 2**: ⏸️ 待开始 (0/8)
- 🏭 **阶段 3**: ⏸️ 待开始 (0/6)
- ⚙️ **阶段 4**: ⏸️ 待开始 (0/6)
- 📊 **阶段 5**: ⏸️ 待开始 (0/10)
- 🔌 **阶段 6**: ⏸️ 待开始 (0/7)
- 🔧 **阶段 7**: ⏸️ 待开始 (0/6)
- 📦 **阶段 8**: ⏸️ 待开始 (0/8)

### 当前任务状态
**正在进行**: 阶段1 - 核心抽象层构建
- **下一步**: 创建 `src/core/base/component.py` - FederatedComponent基类
- **预计完成**: 2025-01-05

---

## 📝 重构日志

### 2024-12-31
- ✅ 创建重构指导文档
- ✅ 完成现有架构深度分析
- ✅ 设计完整的OOP目标架构
- ✅ 制定8阶段重构实施计划
- 📝 **下次更新**: 完成阶段1第一个里程碑时

---

## 🎯 成功标准

### 功能完整性
- [ ] 所有现有功能保持不变
- [ ] 性能不低于当前版本
- [ ] API向后兼容

### 架构质量
- [ ] SOLID原则全面应用
- [ ] 设计模式合理使用
- [ ] 代码耦合度显著降低

### 可维护性
- [ ] 新增功能开发效率提升50%
- [ ] 代码可读性显著改善
- [ ] 测试覆盖率达到90%以上

### 可扩展性
- [ ] 支持插件化扩展
- [ ] 新增数据集/模型/算法容易集成
- [ ] 配置驱动的组件创建

---

## 📖 相关文档

- [CLAUDE.md](./CLAUDE.md) - 框架使用指南
- [README.md](./README.md) - 项目概述
- [requirements.txt](./requirements.txt) - 依赖管理

---

**最后更新**: 2024-12-31  
**下次计划更新**: 2025-01-05 (阶段1里程碑完成时)  
**重构责任人**: Claude Code AI Assistant