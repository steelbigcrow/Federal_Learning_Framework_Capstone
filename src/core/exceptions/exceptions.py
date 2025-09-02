"""
联邦学习框架自定义异常体系

定义了框架中所有自定义异常类，提供详细的错误信息和处理机制。
"""

from typing import Optional, Dict, Any
import traceback
from enum import Enum


class ErrorSeverity(Enum):
    """错误严重程度枚举"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """错误类别枚举"""
    CONFIGURATION = "configuration"
    NETWORK = "network"
    DATA = "data"
    MODEL = "model"
    TRAINING = "training"
    AGGREGATION = "aggregation"
    PLUGIN = "plugin"
    SECURITY = "security"
    RESOURCE = "resource"
    SYSTEM = "system"


class FederatedLearningException(Exception):
    """
    联邦学习框架基础异常类
    
    所有框架特定异常的基类，提供统一的错误处理和报告机制。
    """
    
    def __init__(self, 
                 message: str,
                 error_code: Optional[str] = None,
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 category: ErrorCategory = ErrorCategory.SYSTEM,
                 context: Optional[Dict[str, Any]] = None,
                 cause: Optional[Exception] = None):
        """
        初始化异常
        
        Args:
            message: 错误消息
            error_code: 错误代码
            severity: 错误严重程度
            category: 错误类别
            context: 错误上下文信息
            cause: 引起此异常的原因异常
        """
        super().__init__(message)
        self.message = message
        self.severity = severity
        self.category = category
        self.context = context or {}
        self.cause = cause
        self.error_code = error_code or self._generate_error_code()
        self.traceback_info = traceback.format_exc() if cause else None
        
    def _generate_error_code(self) -> str:
        """生成默认错误代码"""
        return f"{self.category.value.upper()}_{self.__class__.__name__.upper()}"
        
    def get_full_message(self) -> str:
        """获取完整的错误消息"""
        msg = f"[{self.error_code}] {self.message}"
        
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            msg += f" | Context: {context_str}"
            
        if self.cause:
            msg += f" | Caused by: {str(self.cause)}"
            
        return msg
        
    def to_dict(self) -> Dict[str, Any]:
        """将异常转换为字典"""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "severity": self.severity.value,
            "category": self.category.value,
            "context": self.context,
            "cause": str(self.cause) if self.cause else None,
            "traceback": self.traceback_info
        }
        
    def __str__(self) -> str:
        return self.get_full_message()


# =============================================================================
# 配置相关异常
# =============================================================================

class ConfigurationError(FederatedLearningException):
    """配置相关错误基类"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.CONFIGURATION,
            **kwargs
        )


class ConfigValidationError(ConfigurationError):
    """配置验证错误"""
    pass


class ConfigMissingError(ConfigurationError):
    """配置缺失错误"""
    pass


class ConfigTypeError(ConfigurationError):
    """配置类型错误"""
    pass


# =============================================================================
# 客户端相关异常
# =============================================================================

class ClientError(FederatedLearningException):
    """客户端相关错误基类"""
    
    def __init__(self, message: str, client_id: Optional[int] = None, **kwargs):
        context = kwargs.pop('context', {})
        if client_id is not None:
            context['client_id'] = client_id
            
        super().__init__(
            message,
            category=ErrorCategory.TRAINING,
            context=context,
            **kwargs
        )


class ClientConfigurationError(ClientError):
    """客户端配置错误"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, severity=ErrorSeverity.HIGH, **kwargs)


class ClientTrainingError(ClientError):
    """客户端训练错误"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, severity=ErrorSeverity.MEDIUM, **kwargs)


class ClientDataError(ClientError):
    """客户端数据错误"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message, 
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.DATA,
            **kwargs
        )


class ClientNetworkError(ClientError):
    """客户端网络错误"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.NETWORK,
            **kwargs
        )


# =============================================================================
# 服务器相关异常
# =============================================================================

class ServerError(FederatedLearningException):
    """服务器相关错误基类"""
    
    def __init__(self, message: str, **kwargs):
        # Don't set category here to allow subclasses to override it
        super().__init__(message, **kwargs)


class ServerConfigurationError(ServerError):
    """服务器配置错误"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.CRITICAL,
            category=ErrorCategory.CONFIGURATION,
            **kwargs
        )


class ServerOperationError(ServerError):
    """服务器操作错误"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, severity=ErrorSeverity.HIGH, **kwargs)


class ServerResourceError(ServerError):
    """服务器资源错误"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.RESOURCE,
            **kwargs
        )


# =============================================================================
# 聚合相关异常
# =============================================================================

class AggregationError(FederatedLearningException):
    """聚合相关错误基类"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.AGGREGATION,
            **kwargs
        )


class AggregationConfigurationError(AggregationError):
    """聚合配置错误"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.CONFIGURATION,
            **kwargs
        )


class ModelIncompatibilityError(AggregationError):
    """模型不兼容错误"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.MODEL,
            **kwargs
        )


class WeightComputationError(AggregationError):
    """权重计算错误"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, severity=ErrorSeverity.MEDIUM, **kwargs)


# =============================================================================
# 模型相关异常
# =============================================================================

class ModelError(FederatedLearningException):
    """模型相关错误基类"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.MODEL,
            **kwargs
        )


class ModelCreationError(ModelError):
    """模型创建错误"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, severity=ErrorSeverity.HIGH, **kwargs)


class ModelLoadingError(ModelError):
    """模型加载错误"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, severity=ErrorSeverity.HIGH, **kwargs)


class ModelValidationError(ModelError):
    """模型验证错误"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, severity=ErrorSeverity.MEDIUM, **kwargs)


# =============================================================================
# 数据相关异常
# =============================================================================

class DataError(FederatedLearningException):
    """数据相关错误基类"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.DATA,
            **kwargs
        )


class DatasetError(DataError):
    """数据集错误"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, severity=ErrorSeverity.HIGH, **kwargs)


class DataLoaderError(DataError):
    """数据加载器错误"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, severity=ErrorSeverity.MEDIUM, **kwargs)


class DataPartitionError(DataError):
    """数据分割错误"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, severity=ErrorSeverity.HIGH, **kwargs)


class DataValidationError(DataError):
    """数据验证错误"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, severity=ErrorSeverity.MEDIUM, **kwargs)


# =============================================================================
# 策略相关异常
# =============================================================================

class StrategyError(FederatedLearningException):
    """策略相关错误基类"""
    
    def __init__(self, message: str, strategy_name: Optional[str] = None, **kwargs):
        context = kwargs.pop('context', {})
        if strategy_name:
            context['strategy_name'] = strategy_name
            
        super().__init__(
            message,
            category=ErrorCategory.SYSTEM,
            context=context,
            **kwargs
        )


class StrategyNotFoundError(StrategyError):
    """策略未找到错误"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, severity=ErrorSeverity.HIGH, **kwargs)


class StrategyExecutionError(StrategyError):
    """策略执行错误"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, severity=ErrorSeverity.MEDIUM, **kwargs)


# =============================================================================
# 插件相关异常
# =============================================================================

class PluginError(FederatedLearningException):
    """插件相关错误基类"""
    
    def __init__(self, message: str, plugin_name: Optional[str] = None, **kwargs):
        context = kwargs.pop('context', {})
        if plugin_name:
            context['plugin_name'] = plugin_name
            
        super().__init__(
            message,
            category=ErrorCategory.PLUGIN,
            context=context,
            **kwargs
        )


class PluginLoadError(PluginError):
    """插件加载错误"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, severity=ErrorSeverity.HIGH, **kwargs)


class PluginConfigurationError(PluginError):
    """插件配置错误"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.CONFIGURATION,
            **kwargs
        )


class PluginDependencyError(PluginError):
    """插件依赖错误"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, severity=ErrorSeverity.HIGH, **kwargs)


# =============================================================================
# 工厂相关异常
# =============================================================================

class FactoryError(FederatedLearningException):
    """工厂相关错误基类"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.SYSTEM,
            **kwargs
        )


class ComponentCreationError(FactoryError):
    """组件创建错误"""
    
    def __init__(self, message: str, component_type: Optional[str] = None, **kwargs):
        context = kwargs.pop('context', {})
        if component_type:
            context['component_type'] = component_type
            
        super().__init__(
            message,
            severity=ErrorSeverity.HIGH,
            context=context,
            **kwargs
        )


class FactoryRegistrationError(FactoryError):
    """工厂注册错误"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, severity=ErrorSeverity.MEDIUM, **kwargs)


# =============================================================================
# 资源相关异常
# =============================================================================

class ResourceError(FederatedLearningException):
    """资源相关错误基类"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.RESOURCE,
            **kwargs
        )


class OutOfMemoryError(ResourceError):
    """内存不足错误"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, severity=ErrorSeverity.CRITICAL, **kwargs)


class DiskSpaceError(ResourceError):
    """磁盘空间不足错误"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, severity=ErrorSeverity.HIGH, **kwargs)


class ComputeResourceError(ResourceError):
    """计算资源错误"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, severity=ErrorSeverity.MEDIUM, **kwargs)


# =============================================================================
# 安全相关异常
# =============================================================================

class SecurityError(FederatedLearningException):
    """安全相关错误基类"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.CRITICAL,
            category=ErrorCategory.SECURITY,
            **kwargs
        )


class AuthenticationError(SecurityError):
    """认证错误"""
    pass


class AuthorizationError(SecurityError):
    """授权错误"""
    pass


class DataPrivacyError(SecurityError):
    """数据隐私错误"""
    pass


# =============================================================================
# 异常处理工具类
# =============================================================================

class ExceptionHandler:
    """
    异常处理工具类
    
    提供统一的异常处理、记录和报告功能。
    """
    
    @staticmethod
    def handle_exception(exception: Exception, 
                        logger=None, 
                        reraise: bool = True,
                        context: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        处理异常
        
        Args:
            exception: 异常实例
            logger: 日志记录器
            reraise: 是否重新抛出异常
            context: 额外的上下文信息
            
        Returns:
            异常信息字典（如果不重新抛出）
        """
        # 创建标准化的异常信息
        if isinstance(exception, FederatedLearningException):
            exc_info = exception.to_dict()
        else:
            # 将普通异常包装为框架异常
            wrapped_exc = FederatedLearningException(
                str(exception),
                cause=exception,
                context=context
            )
            exc_info = wrapped_exc.to_dict()
            
        # 记录异常
        if logger:
            logger.error(f"Exception occurred: {exc_info}")
            
        if reraise:
            raise exception
        else:
            return exc_info
            
    @staticmethod
    def create_error_response(exception: Exception) -> Dict[str, Any]:
        """
        创建错误响应
        
        Args:
            exception: 异常实例
            
        Returns:
            标准化的错误响应字典
        """
        if isinstance(exception, FederatedLearningException):
            return {
                "success": False,
                "error": exception.to_dict()
            }
        else:
            return {
                "success": False,
                "error": {
                    "error_code": "UNKNOWN_ERROR",
                    "message": str(exception),
                    "severity": ErrorSeverity.MEDIUM.value,
                    "category": ErrorCategory.SYSTEM.value
                }
            }