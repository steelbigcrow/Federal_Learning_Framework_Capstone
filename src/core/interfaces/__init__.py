"""
核心接口模块

导出所有核心接口定义，提供统一的接口访问。
"""

from .factory import (
    FactoryInterface,
    ModelFactoryInterface,
    DatasetFactoryInterface, 
    ClientFactoryInterface,
    ServerFactoryInterface,
    AggregatorFactoryInterface,
    ComponentFactoryInterface,
    ComponentType,
    factory_register
)

from .strategy import (
    StrategyInterface,
    AggregationStrategyInterface,
    TrainingStrategyInterface,
    DataPartitionStrategyInterface,
    ClientSelectionStrategyInterface,
    OptimizationStrategyInterface,
    EvaluationStrategyInterface,
    StrategyRegistry,
    StrategyType,
    strategy_register
)

from .plugin import (
    PluginInterface,
    FeatureExtensionPlugin,
    AggregationAlgorithmPlugin,
    MetricCollectorPlugin,
    VisualizationPlugin,
    PluginManager,
    PluginType,
    PluginLifecycle,
    PluginError,
    PluginConfigurationError,
    PluginDependencyError
)

__all__ = [
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
    'PluginError',
    'PluginConfigurationError',
    'PluginDependencyError',
]