"""
插件接口定义

定义了插件系统的标准接口，支持框架的模块化扩展。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
import inspect


class PluginType(Enum):
    """插件类型枚举"""
    FEATURE_EXTENSION = "feature_extension"  # 功能扩展插件（如LoRA、AdaLoRA）
    AGGREGATION_ALGORITHM = "aggregation_algorithm"  # 聚合算法插件
    DATA_PROCESSOR = "data_processor"  # 数据处理插件
    METRIC_COLLECTOR = "metric_collector"  # 指标收集插件
    VISUALIZATION = "visualization"  # 可视化插件
    COMMUNICATION = "communication"  # 通信协议插件
    SECURITY = "security"  # 安全相关插件
    CUSTOM = "custom"  # 自定义插件


class PluginLifecycle(Enum):
    """插件生命周期状态枚举"""
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    INITIALIZING = "initializing"
    ACTIVE = "active"
    PAUSED = "paused"
    ERROR = "error"
    UNLOADING = "unloading"


class PluginInterface(ABC):
    """
    插件接口基类
    
    定义了所有插件必须实现的基本接口和生命周期管理。
    
    设计原则：
    - 插件化架构：支持运行时加载和卸载
    - 生命周期管理：标准化的初始化和清理流程
    - 依赖管理：支持插件间依赖关系
    - 配置管理：插件特定的配置支持
    """
    
    def __init__(self):
        self._status = PluginLifecycle.UNLOADED
        self._config: Dict[str, Any] = {}
        self._dependencies: List[str] = []
        self._hooks: Dict[str, List[Callable]] = {}
        
    @abstractmethod
    def get_name(self) -> str:
        """获取插件名称"""
        pass
        
    @abstractmethod
    def get_version(self) -> str:
        """获取插件版本"""
        pass
        
    @abstractmethod
    def get_description(self) -> str:
        """获取插件描述"""
        pass
        
    @abstractmethod
    def get_plugin_type(self) -> PluginType:
        """获取插件类型"""
        pass
        
    @abstractmethod
    def load(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        加载插件
        
        Args:
            config: 插件配置
            
        Raises:
            PluginError: 当加载失败时
        """
        pass
        
    @abstractmethod
    def unload(self) -> None:
        """
        卸载插件
        
        Raises:
            PluginError: 当卸载失败时
        """
        pass
        
    @abstractmethod
    def initialize(self, context: Dict[str, Any]) -> None:
        """
        初始化插件
        
        Args:
            context: 初始化上下文
            
        Raises:
            PluginError: 当初始化失败时
        """
        pass
        
    @abstractmethod
    def cleanup(self) -> None:
        """
        清理插件资源
        
        Raises:
            PluginError: 当清理失败时
        """
        pass
        
    def get_status(self) -> PluginLifecycle:
        """获取插件状态"""
        return self._status
        
    def get_config(self) -> Dict[str, Any]:
        """获取插件配置"""
        return self._config.copy()
        
    def get_dependencies(self) -> List[str]:
        """获取插件依赖列表"""
        return self._dependencies.copy()
        
    def get_config_schema(self) -> Dict[str, Any]:
        """
        获取插件配置模式
        
        Returns:
            JSON Schema格式的配置模式
        """
        return {}
        
    def validate_config(self, config: Dict[str, Any]) -> None:
        """
        验证插件配置
        
        Args:
            config: 待验证的配置
            
        Raises:
            PluginConfigurationError: 当配置无效时
        """
        # 默认实现：不验证
        pass
        
    def register_hook(self, event: str, callback: Callable) -> None:
        """
        注册事件钩子
        
        Args:
            event: 事件名称
            callback: 回调函数
        """
        if event not in self._hooks:
            self._hooks[event] = []
        self._hooks[event].append(callback)
        
    def unregister_hook(self, event: str, callback: Callable) -> None:
        """取消注册事件钩子"""
        if event in self._hooks and callback in self._hooks[event]:
            self._hooks[event].remove(callback)
            
    def trigger_hook(self, event: str, *args, **kwargs) -> List[Any]:
        """
        触发事件钩子
        
        Args:
            event: 事件名称
            *args, **kwargs: 传递给钩子的参数
            
        Returns:
            所有钩子的返回值列表
        """
        results = []
        for callback in self._hooks.get(event, []):
            try:
                result = callback(*args, **kwargs)
                results.append(result)
            except Exception as e:
                # 记录错误但不中断其他钩子的执行
                print(f"Hook execution failed for {event}: {e}")
        return results
        
    def _set_status(self, status: PluginLifecycle) -> None:
        """设置插件状态"""
        self._status = status
        
    def __str__(self) -> str:
        """字符串表示"""
        return f"{self.get_name()} v{self.get_version()} ({self._status.value})"


class FeatureExtensionPlugin(PluginInterface):
    """
    功能扩展插件接口
    
    专门用于扩展框架功能的插件，如LoRA、AdaLoRA等。
    """
    
    @abstractmethod
    def apply_to_model(self, model: Any, config: Dict[str, Any]) -> Any:
        """
        将功能应用到模型
        
        Args:
            model: 目标模型
            config: 应用配置
            
        Returns:
            修改后的模型
        """
        pass
        
    @abstractmethod
    def remove_from_model(self, model: Any) -> Any:
        """
        从模型中移除功能
        
        Args:
            model: 目标模型
            
        Returns:
            恢复的模型
        """
        pass
        
    def is_compatible_with_model(self, model: Any) -> bool:
        """
        检查是否与模型兼容
        
        Args:
            model: 目标模型
            
        Returns:
            是否兼容
        """
        return True
        
    def get_required_parameters(self) -> List[str]:
        """获取必需的配置参数列表"""
        return []


class AggregationAlgorithmPlugin(PluginInterface):
    """
    聚合算法插件接口
    
    专门用于实现自定义聚合算法的插件。
    """
    
    @abstractmethod
    def aggregate(self, 
                  client_models: List[Dict[str, Any]],
                  client_weights: List[float],
                  config: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行模型聚合
        
        Args:
            client_models: 客户端模型列表
            client_weights: 客户端权重列表
            config: 聚合配置
            
        Returns:
            聚合后的模型
        """
        pass
        
    def get_algorithm_name(self) -> str:
        """获取算法名称"""
        return self.get_name()
        
    def supports_secure_aggregation(self) -> bool:
        """是否支持安全聚合"""
        return False


class MetricCollectorPlugin(PluginInterface):
    """
    指标收集插件接口
    
    专门用于收集和处理训练指标的插件。
    """
    
    @abstractmethod
    def collect_metrics(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        收集指标
        
        Args:
            context: 收集上下文
            
        Returns:
            收集到的指标
        """
        pass
        
    @abstractmethod
    def process_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理指标
        
        Args:
            metrics: 原始指标
            
        Returns:
            处理后的指标
        """
        pass
        
    def get_supported_metrics(self) -> List[str]:
        """获取支持的指标列表"""
        return []


class VisualizationPlugin(PluginInterface):
    """
    可视化插件接口
    
    专门用于生成可视化图表和报告的插件。
    """
    
    @abstractmethod
    def create_visualization(self, 
                           data: Dict[str, Any], 
                           config: Dict[str, Any]) -> Any:
        """
        创建可视化
        
        Args:
            data: 可视化数据
            config: 可视化配置
            
        Returns:
            可视化对象或文件路径
        """
        pass
        
    def get_supported_formats(self) -> List[str]:
        """获取支持的输出格式"""
        return ["png", "pdf", "svg"]
        
    def get_supported_chart_types(self) -> List[str]:
        """获取支持的图表类型"""
        return ["line", "bar", "scatter"]


class PluginManager:
    """
    插件管理器
    
    负责插件的发现、加载、管理和生命周期控制。
    """
    
    def __init__(self):
        self._plugins: Dict[str, PluginInterface] = {}
        self._plugin_types: Dict[PluginType, List[str]] = {}
        self._dependency_graph: Dict[str, List[str]] = {}
        
    def register_plugin(self, plugin: PluginInterface) -> None:
        """
        注册插件
        
        Args:
            plugin: 插件实例
            
        Raises:
            PluginError: 当注册失败时
        """
        plugin_name = plugin.get_name()
        
        if plugin_name in self._plugins:
            raise PluginError(f"Plugin '{plugin_name}' is already registered")
            
        self._plugins[plugin_name] = plugin
        
        # 按类型分类
        plugin_type = plugin.get_plugin_type()
        if plugin_type not in self._plugin_types:
            self._plugin_types[plugin_type] = []
        self._plugin_types[plugin_type].append(plugin_name)
        
        # 记录依赖关系
        self._dependency_graph[plugin_name] = plugin.get_dependencies()
        
    def unregister_plugin(self, plugin_name: str) -> None:
        """取消注册插件"""
        if plugin_name in self._plugins:
            plugin = self._plugins[plugin_name]
            
            # 卸载插件
            if plugin.get_status() != PluginLifecycle.UNLOADED:
                self.unload_plugin(plugin_name)
                
            # 从注册表中移除
            del self._plugins[plugin_name]
            
            # 从类型分类中移除
            plugin_type = plugin.get_plugin_type()
            if plugin_name in self._plugin_types.get(plugin_type, []):
                self._plugin_types[plugin_type].remove(plugin_name)
                
            # 移除依赖关系
            self._dependency_graph.pop(plugin_name, None)
            
    def load_plugin(self, plugin_name: str, config: Optional[Dict[str, Any]] = None) -> None:
        """
        加载插件
        
        Args:
            plugin_name: 插件名称
            config: 插件配置
        """
        plugin = self.get_plugin(plugin_name)
        if plugin is None:
            raise PluginError(f"Plugin '{plugin_name}' not found")
            
        # 检查依赖
        self._check_dependencies(plugin_name)
        
        plugin.load(config)
        
    def unload_plugin(self, plugin_name: str) -> None:
        """卸载插件"""
        plugin = self.get_plugin(plugin_name)
        if plugin is not None:
            plugin.unload()
            
    def get_plugin(self, plugin_name: str) -> Optional[PluginInterface]:
        """获取插件实例"""
        return self._plugins.get(plugin_name)
        
    def list_plugins(self, plugin_type: Optional[PluginType] = None) -> List[str]:
        """
        列出插件
        
        Args:
            plugin_type: 插件类型过滤器
            
        Returns:
            插件名称列表
        """
        if plugin_type is None:
            return list(self._plugins.keys())
        return self._plugin_types.get(plugin_type, [])
        
    def get_plugins_by_type(self, plugin_type: PluginType) -> List[PluginInterface]:
        """获取指定类型的所有插件实例"""
        plugin_names = self._plugin_types.get(plugin_type, [])
        return [self._plugins[name] for name in plugin_names if name in self._plugins]
        
    def _check_dependencies(self, plugin_name: str) -> None:
        """检查插件依赖"""
        dependencies = self._dependency_graph.get(plugin_name, [])
        
        for dep in dependencies:
            if dep not in self._plugins:
                raise PluginError(f"Dependency '{dep}' not found for plugin '{plugin_name}'")
                
            dep_plugin = self._plugins[dep]
            if dep_plugin.get_status() not in [PluginLifecycle.LOADED, PluginLifecycle.ACTIVE]:
                raise PluginError(f"Dependency '{dep}' is not loaded for plugin '{plugin_name}'")


# 插件错误类
class PluginError(Exception):
    """插件相关错误"""
    pass


class PluginConfigurationError(PluginError):
    """插件配置错误"""
    pass


class PluginDependencyError(PluginError):
    """插件依赖错误"""
    pass