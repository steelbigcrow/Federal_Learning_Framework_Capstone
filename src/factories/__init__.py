"""
工厂模式实现模块

实现了联邦学习框架的工厂模式，用于统一管理组件创建和实例化。
包含以下核心组件：

- ComponentFactory: 主工厂，统一管理所有子工厂
- ModelFactory: 模型创建工厂  
- DatasetFactory: 数据集创建工厂
- ClientFactory: 客户端创建工厂
- ServerFactory: 服务器创建工厂
- FactoryRegistry: 工厂注册管理系统

设计原则：
- 抽象工厂模式：提供统一的组件创建接口
- 单一职责：每个工厂专注于特定类型的组件
- 开闭原则：通过注册机制支持动态扩展
- 依赖注入：支持配置驱动的组件创建
"""

from .component_factory import ComponentFactory
from .model_factory import ModelFactory  
from .dataset_factory import DatasetFactory
from .client_factory import ClientFactory
from .server_factory import ServerFactory
from .factory_registry import FactoryRegistry

__all__ = [
    'ComponentFactory',
    'ModelFactory', 
    'DatasetFactory',
    'ClientFactory',
    'ServerFactory', 
    'FactoryRegistry'
]