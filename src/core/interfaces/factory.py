"""
工厂接口定义

定义了工厂模式的标准接口，用于统一组件创建和管理。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, TypeVar, Generic
from enum import Enum


T = TypeVar('T')  # 泛型类型变量


class ComponentType(Enum):
    """组件类型枚举"""
    CLIENT = "client"
    SERVER = "server"
    AGGREGATOR = "aggregator"
    MODEL = "model"
    DATASET = "dataset"
    STRATEGY = "strategy"
    PLUGIN = "plugin"


class FactoryInterface(ABC, Generic[T]):
    """
    工厂接口基类
    
    定义了所有工厂类必须实现的基本接口，支持泛型以提供类型安全。
    
    设计原则：
    - 抽象工厂模式：定义创建相关对象的接口
    - 单一职责：专注于对象创建
    - 开闭原则：通过注册机制支持扩展
    """
    
    @abstractmethod
    def create(self, component_type: str, config: Dict[str, Any], **kwargs) -> T:
        """
        创建组件实例
        
        Args:
            component_type: 组件类型标识
            config: 组件配置
            **kwargs: 额外的创建参数
            
        Returns:
            创建的组件实例
            
        Raises:
            FactoryError: 当创建失败时
        """
        pass
        
    @abstractmethod
    def register(self, component_type: str, creator_class: Type[T]) -> None:
        """
        注册组件创建器
        
        Args:
            component_type: 组件类型标识
            creator_class: 创建器类
            
        Raises:
            FactoryError: 当注册失败时
        """
        pass
        
    @abstractmethod
    def unregister(self, component_type: str) -> None:
        """
        取消注册组件创建器
        
        Args:
            component_type: 组件类型标识
        """
        pass
        
    @abstractmethod
    def list_registered_types(self) -> List[str]:
        """
        列出所有已注册的组件类型
        
        Returns:
            已注册组件类型列表
        """
        pass
        
    @abstractmethod
    def is_registered(self, component_type: str) -> bool:
        """
        检查组件类型是否已注册
        
        Args:
            component_type: 组件类型标识
            
        Returns:
            是否已注册
        """
        pass


class ModelFactoryInterface(FactoryInterface):
    """
    模型工厂接口
    
    专门用于创建机器学习模型的工厂接口。
    """
    
    @abstractmethod
    def create_model(self, 
                     dataset: str, 
                     model_type: str, 
                     config: Dict[str, Any],
                     **kwargs) -> Any:
        """
        创建模型实例
        
        Args:
            dataset: 数据集名称
            model_type: 模型类型
            config: 模型配置
            **kwargs: 额外参数（如vocab_size等）
            
        Returns:
            创建的模型实例
        """
        pass
        
    @abstractmethod
    def get_supported_datasets(self) -> List[str]:
        """获取支持的数据集列表"""
        pass
        
    @abstractmethod
    def get_supported_models(self, dataset: str) -> List[str]:
        """获取指定数据集支持的模型列表"""
        pass


class DatasetFactoryInterface(FactoryInterface):
    """
    数据集工厂接口
    
    专门用于创建和管理数据集的工厂接口。
    """
    
    @abstractmethod
    def create_dataset(self, 
                       dataset_name: str, 
                       split: str,
                       config: Dict[str, Any]) -> Any:
        """
        创建数据集实例
        
        Args:
            dataset_name: 数据集名称
            split: 数据分割类型（train/test/val）
            config: 数据集配置
            
        Returns:
            创建的数据集实例
        """
        pass
        
    @abstractmethod
    def create_dataloader(self, 
                          dataset: Any, 
                          config: Dict[str, Any]) -> Any:
        """
        创建数据加载器
        
        Args:
            dataset: 数据集实例
            config: 数据加载器配置
            
        Returns:
            创建的数据加载器实例
        """
        pass
        
    @abstractmethod
    def partition_dataset(self, 
                          dataset: Any, 
                          num_clients: int,
                          partition_strategy: str,
                          config: Dict[str, Any]) -> List[Any]:
        """
        分割数据集为多个客户端数据
        
        Args:
            dataset: 原始数据集
            num_clients: 客户端数量
            partition_strategy: 分割策略
            config: 分割配置
            
        Returns:
            分割后的数据集列表
        """
        pass


class ClientFactoryInterface(FactoryInterface):
    """
    客户端工厂接口
    
    专门用于创建联邦学习客户端的工厂接口。
    """
    
    @abstractmethod
    def create_client(self, 
                      client_id: int,
                      train_dataloader: Any,
                      config: Dict[str, Any]) -> Any:
        """
        创建客户端实例
        
        Args:
            client_id: 客户端ID
            train_dataloader: 训练数据加载器
            config: 客户端配置
            
        Returns:
            创建的客户端实例
        """
        pass
        
    @abstractmethod
    def create_clients_batch(self, 
                            dataloaders: List[Any],
                            config: Dict[str, Any]) -> List[Any]:
        """
        批量创建客户端
        
        Args:
            dataloaders: 数据加载器列表
            config: 客户端配置
            
        Returns:
            创建的客户端列表
        """
        pass


class ServerFactoryInterface(FactoryInterface):
    """
    服务器工厂接口
    
    专门用于创建联邦学习服务器的工厂接口。
    """
    
    @abstractmethod
    def create_server(self, 
                      model_constructor: Any,
                      clients: List[Any],
                      config: Dict[str, Any]) -> Any:
        """
        创建服务器实例
        
        Args:
            model_constructor: 模型构造函数
            clients: 客户端列表
            config: 服务器配置
            
        Returns:
            创建的服务器实例
        """
        pass


class AggregatorFactoryInterface(FactoryInterface):
    """
    聚合器工厂接口
    
    专门用于创建模型聚合器的工厂接口。
    """
    
    @abstractmethod
    def create_aggregator(self, 
                          aggregation_strategy: str,
                          config: Dict[str, Any]) -> Any:
        """
        创建聚合器实例
        
        Args:
            aggregation_strategy: 聚合策略名称
            config: 聚合器配置
            
        Returns:
            创建的聚合器实例
        """
        pass
        
    @abstractmethod
    def get_available_strategies(self) -> List[str]:
        """获取可用的聚合策略列表"""
        pass


class ComponentFactoryInterface(ABC):
    """
    主工厂接口
    
    统一管理所有子工厂，提供一站式的组件创建服务。
    """
    
    @abstractmethod
    def get_model_factory(self) -> ModelFactoryInterface:
        """获取模型工厂"""
        pass
        
    @abstractmethod
    def get_dataset_factory(self) -> DatasetFactoryInterface:
        """获取数据集工厂"""
        pass
        
    @abstractmethod
    def get_client_factory(self) -> ClientFactoryInterface:
        """获取客户端工厂"""
        pass
        
    @abstractmethod
    def get_server_factory(self) -> ServerFactoryInterface:
        """获取服务器工厂"""
        pass
        
    @abstractmethod
    def get_aggregator_factory(self) -> AggregatorFactoryInterface:
        """获取聚合器工厂"""
        pass
        
    @abstractmethod
    def create_federated_setup(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        创建完整的联邦学习设置
        
        Args:
            config: 完整的配置字典
            
        Returns:
            包含所有组件的字典
        """
        pass


# 工厂注册装饰器
def factory_register(factory_type: ComponentType, component_name: str):
    """
    工厂注册装饰器
    
    用于自动注册组件创建器到对应的工厂。
    
    Args:
        factory_type: 工厂类型
        component_name: 组件名称
    
    Example:
        @factory_register(ComponentType.MODEL, "mnist_mlp")
        class MnistMLPCreator:
            def create(self, config):
                return MnistMLP(**config)
    """
    def decorator(cls):
        # 在实际实现中，这里会将类注册到对应的工厂
        cls._factory_type = factory_type
        cls._component_name = component_name
        return cls
    return decorator