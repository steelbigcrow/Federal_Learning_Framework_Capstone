"""
异常模块

导出所有自定义异常类，提供统一的异常处理。
"""

from .exceptions import (
    # 基础异常类
    FederatedLearningException,
    ErrorSeverity,
    ErrorCategory,
    
    # 配置异常
    ConfigurationError,
    ConfigValidationError,
    ConfigMissingError,
    ConfigTypeError,
    
    # 客户端异常
    ClientError,
    ClientConfigurationError,
    ClientTrainingError,
    ClientDataError,
    ClientNetworkError,
    
    # 服务器异常
    ServerError,
    ServerConfigurationError,
    ServerOperationError,
    ServerResourceError,
    
    # 聚合异常
    AggregationError,
    AggregationConfigurationError,
    ModelIncompatibilityError,
    WeightComputationError,
    
    # 模型异常
    ModelError,
    ModelCreationError,
    ModelLoadingError,
    ModelValidationError,
    
    # 数据异常
    DataError,
    DatasetError,
    DataLoaderError,
    DataPartitionError,
    DataValidationError,
    
    # 策略异常
    StrategyError,
    StrategyNotFoundError,
    StrategyExecutionError,
    
    # 插件异常
    PluginError,
    PluginLoadError,
    PluginConfigurationError,
    PluginDependencyError,
    
    # 工厂异常
    FactoryError,
    ComponentCreationError,
    FactoryRegistrationError,
    
    # 资源异常
    ResourceError,
    OutOfMemoryError,
    DiskSpaceError,
    ComputeResourceError,
    
    # 安全异常
    SecurityError,
    AuthenticationError,
    AuthorizationError,
    DataPrivacyError,
    
    # 工具类
    ExceptionHandler,
)

__all__ = [
    # 基础异常
    'FederatedLearningException',
    'ErrorSeverity',
    'ErrorCategory',
    
    # 配置异常
    'ConfigurationError',
    'ConfigValidationError', 
    'ConfigMissingError',
    'ConfigTypeError',
    
    # 客户端异常
    'ClientError',
    'ClientConfigurationError',
    'ClientTrainingError',
    'ClientDataError',
    'ClientNetworkError',
    
    # 服务器异常
    'ServerError',
    'ServerConfigurationError',
    'ServerOperationError',
    'ServerResourceError',
    
    # 聚合异常
    'AggregationError',
    'AggregationConfigurationError',
    'ModelIncompatibilityError',
    'WeightComputationError',
    
    # 模型异常
    'ModelError',
    'ModelCreationError',
    'ModelLoadingError',
    'ModelValidationError',
    
    # 数据异常
    'DataError',
    'DatasetError',
    'DataLoaderError',
    'DataPartitionError', 
    'DataValidationError',
    
    # 策略异常
    'StrategyError',
    'StrategyNotFoundError',
    'StrategyExecutionError',
    
    # 插件异常
    'PluginError',
    'PluginLoadError',
    'PluginConfigurationError',
    'PluginDependencyError',
    
    # 工厂异常
    'FactoryError',
    'ComponentCreationError',
    'FactoryRegistrationError',
    
    # 资源异常
    'ResourceError',
    'OutOfMemoryError',
    'DiskSpaceError',
    'ComputeResourceError',
    
    # 安全异常
    'SecurityError',
    'AuthenticationError',
    'AuthorizationError',
    'DataPrivacyError',
    
    # 工具类
    'ExceptionHandler',
]