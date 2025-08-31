"""
核心抽象层模块

联邦学习框架的核心抽象层，提供所有基础组件的抽象定义。

该模块包含：
- 基础组件抽象类
- 核心接口定义  
- 异常体系
- 设计模式支持

使用示例：
    from src.core import AbstractClient, AbstractServer, AbstractAggregator
    from src.core import FactoryInterface, StrategyInterface
    from src.core import FederatedLearningException
"""

# 导入基础组件
from .base import (
    FederatedComponent,
    ComponentStatus,
    ComponentRegistry,
    AbstractClient, 
    AbstractServer,
    AbstractAggregator,
    AggregationMode
)

# 导入接口定义
from .interfaces import (
    # 工厂接口
    FactoryInterface,
    ModelFactoryInterface,
    DatasetFactoryInterface,
    ClientFactoryInterface,
    ServerFactoryInterface, 
    AggregatorFactoryInterface,
    ComponentFactoryInterface,
    ComponentType,
    factory_register,
    
    # 策略接口
    StrategyInterface,
    AggregationStrategyInterface,
    TrainingStrategyInterface,
    DataPartitionStrategyInterface,
    ClientSelectionStrategyInterface,
    OptimizationStrategyInterface,
    EvaluationStrategyInterface,
    StrategyRegistry,
    StrategyType,
    strategy_register,
    
    # 插件接口
    PluginInterface,
    FeatureExtensionPlugin,
    AggregationAlgorithmPlugin,
    MetricCollectorPlugin,
    VisualizationPlugin,
    PluginManager,
    PluginType,
    PluginLifecycle
)

# 导入异常体系
from .exceptions import (
    # 基础异常
    FederatedLearningException,
    ErrorSeverity,
    ErrorCategory,
    
    # 主要异常类
    ConfigurationError,
    ClientError,
    ServerError,
    AggregationError,
    ModelError,
    DataError,
    StrategyError,
    PluginError,
    FactoryError,
    ResourceError,
    SecurityError,
    
    # 异常处理工具
    ExceptionHandler
)

# 版本信息
__version__ = "1.0.0"
__author__ = "Claude Code AI Assistant"

__all__ = [
    # 基础组件
    'FederatedComponent',
    'ComponentStatus',
    'ComponentRegistry',
    'AbstractClient',
    'AbstractServer', 
    'AbstractAggregator',
    'AggregationMode',
    
    # 工厂接口
    'FactoryInterface',
    'ModelFactoryInterface',
    'DatasetFactoryInterface',
    'ClientFactoryInterface',
    'ServerFactoryInterface',
    'AggregatorFactoryInterface', 
    'ComponentFactoryInterface',
    'ComponentType',
    'factory_register',
    
    # 策略接口
    'StrategyInterface',
    'AggregationStrategyInterface',
    'TrainingStrategyInterface',
    'DataPartitionStrategyInterface',
    'ClientSelectionStrategyInterface',
    'OptimizationStrategyInterface',
    'EvaluationStrategyInterface', 
    'StrategyRegistry',
    'StrategyType',
    'strategy_register',
    
    # 插件接口
    'PluginInterface',
    'FeatureExtensionPlugin',
    'AggregationAlgorithmPlugin',
    'MetricCollectorPlugin',
    'VisualizationPlugin',
    'PluginManager',
    'PluginType',
    'PluginLifecycle',
    
    # 异常体系
    'FederatedLearningException',
    'ErrorSeverity',
    'ErrorCategory',
    'ConfigurationError',
    'ClientError',
    'ServerError',
    'AggregationError',
    'ModelError',
    'DataError',
    'StrategyError', 
    'PluginError',
    'FactoryError',
    'ResourceError',
    'SecurityError',
    'ExceptionHandler',
    
    # 元信息
    '__version__',
    '__author__',
]