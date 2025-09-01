"""
工厂注册管理系统

提供工厂的注册、发现和管理功能，支持动态工厂注册和组件创建器管理。
"""

from typing import Dict, List, Type, Any, Optional
from abc import ABC, abstractmethod
import threading
import logging
from ..core.exceptions import FactoryError
from ..core.interfaces.factory import ComponentType, FactoryInterface


class FactoryRegistry:
    """
    工厂注册管理器
    
    提供线程安全的工厂注册和管理功能，支持：
    - 动态工厂注册和注销
    - 工厂实例管理（单例模式）
    - 组件创建器的注册和发现
    - 类型安全的工厂操作
    
    采用单例模式确保全局唯一性。
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
            
        self._factories: Dict[str, Type[FactoryInterface]] = {}
        self._factory_instances: Dict[str, FactoryInterface] = {}
        self._component_creators: Dict[ComponentType, Dict[str, Type]] = {
            component_type: {} for component_type in ComponentType
        }
        self._lock = threading.Lock()
        self._logger = logging.getLogger(__name__)
        self._initialized = True
    
    def register_factory(self, 
                        factory_type: str, 
                        factory_class: Type[FactoryInterface]) -> None:
        """
        注册工厂类
        
        Args:
            factory_type: 工厂类型标识
            factory_class: 工厂类
            
        Raises:
            FactoryError: 当工厂类型已存在时
        """
        with self._lock:
            if factory_type in self._factories:
                raise FactoryError(f"Factory type '{factory_type}' already registered")
                
            # Check if it implements either FactoryInterface or ComponentFactoryInterface
            from ..core.interfaces.factory import ComponentFactoryInterface
            if not (issubclass(factory_class, FactoryInterface) or 
                    issubclass(factory_class, ComponentFactoryInterface)):
                raise FactoryError(
                    f"Factory class must implement FactoryInterface or ComponentFactoryInterface, got {factory_class}"
                )
                
            self._factories[factory_type] = factory_class
            self._logger.info(f"Registered factory: {factory_type}")
    
    def unregister_factory(self, factory_type: str) -> None:
        """
        注销工厂类
        
        Args:
            factory_type: 工厂类型标识
        """
        with self._lock:
            if factory_type in self._factories:
                del self._factories[factory_type]
                
            if factory_type in self._factory_instances:
                del self._factory_instances[factory_type]
                
            self._logger.info(f"Unregistered factory: {factory_type}")
    
    def get_factory(self, factory_type: str, **kwargs) -> FactoryInterface:
        """
        获取工厂实例（单例模式）
        
        Args:
            factory_type: 工厂类型标识
            **kwargs: 工厂初始化参数
            
        Returns:
            工厂实例
            
        Raises:
            FactoryError: 当工厂类型不存在时
        """
        with self._lock:
            if factory_type not in self._factories:
                raise FactoryError(f"Unknown factory type: {factory_type}")
                
            if factory_type not in self._factory_instances:
                factory_class = self._factories[factory_type]
                try:
                    self._factory_instances[factory_type] = factory_class(**kwargs)
                    self._logger.debug(f"Created factory instance: {factory_type}")
                except Exception as e:
                    raise FactoryError(f"Failed to create factory {factory_type}: {e}")
                    
            return self._factory_instances[factory_type]
    
    def register_component_creator(self,
                                  component_type: ComponentType,
                                  name: str,
                                  creator_class: Type) -> None:
        """
        注册组件创建器
        
        Args:
            component_type: 组件类型
            name: 组件名称
            creator_class: 创建器类
            
        Raises:
            FactoryError: 当组件名称已存在时
        """
        with self._lock:
            if name in self._component_creators[component_type]:
                raise FactoryError(
                    f"Component creator '{name}' already registered for type {component_type}"
                )
                
            self._component_creators[component_type][name] = creator_class
            self._logger.info(f"Registered component creator: {component_type}.{name}")
    
    def unregister_component_creator(self,
                                   component_type: ComponentType,
                                   name: str) -> None:
        """
        注销组件创建器
        
        Args:
            component_type: 组件类型
            name: 组件名称
        """
        with self._lock:
            if name in self._component_creators[component_type]:
                del self._component_creators[component_type][name]
                self._logger.info(f"Unregistered component creator: {component_type}.{name}")
    
    def get_component_creator(self,
                             component_type: ComponentType,
                             name: str) -> Optional[Type]:
        """
        获取组件创建器
        
        Args:
            component_type: 组件类型
            name: 组件名称
            
        Returns:
            创建器类，如果不存在返回None
        """
        with self._lock:
            return self._component_creators[component_type].get(name)
    
    def list_registered_factories(self) -> List[str]:
        """
        列出所有已注册的工厂类型
        
        Returns:
            工厂类型列表
        """
        with self._lock:
            return list(self._factories.keys())
    
    def list_component_creators(self, 
                              component_type: ComponentType) -> List[str]:
        """
        列出指定类型的所有组件创建器
        
        Args:
            component_type: 组件类型
            
        Returns:
            组件创建器名称列表
        """
        with self._lock:
            return list(self._component_creators[component_type].keys())
    
    def is_factory_registered(self, factory_type: str) -> bool:
        """
        检查工厂类型是否已注册
        
        Args:
            factory_type: 工厂类型标识
            
        Returns:
            是否已注册
        """
        with self._lock:
            return factory_type in self._factories
    
    def is_component_creator_registered(self,
                                      component_type: ComponentType,
                                      name: str) -> bool:
        """
        检查组件创建器是否已注册
        
        Args:
            component_type: 组件类型
            name: 组件名称
            
        Returns:
            是否已注册
        """
        with self._lock:
            return name in self._component_creators[component_type]
    
    def clear_registry(self) -> None:
        """
        清空所有注册信息（主要用于测试）
        """
        with self._lock:
            self._factories.clear()
            self._factory_instances.clear()
            for component_type in ComponentType:
                self._component_creators[component_type].clear()
            self._logger.info("Cleared factory registry")
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """
        获取注册统计信息
        
        Returns:
            包含注册统计信息的字典
        """
        with self._lock:
            stats = {
                'factory_types': len(self._factories),
                'factory_instances': len(self._factory_instances),
                'component_creators': {
                    component_type.value: len(creators)
                    for component_type, creators in self._component_creators.items()
                },
                'registered_factories': list(self._factories.keys()),
                'total_creators': sum(
                    len(creators) for creators in self._component_creators.values()
                )
            }
            return stats


# 全局工厂注册实例
_factory_registry = FactoryRegistry()


def get_factory_registry() -> FactoryRegistry:
    """
    获取全局工厂注册实例
    
    Returns:
        FactoryRegistry实例
    """
    return _factory_registry


# 装饰器用于自动注册组件创建器
def register_component_creator(component_type: ComponentType, name: str):
    """
    组件创建器注册装饰器
    
    Args:
        component_type: 组件类型
        name: 组件名称
        
    Example:
        @register_component_creator(ComponentType.MODEL, "mnist_mlp")
        class MnistMLPCreator:
            @staticmethod
            def create(config):
                return MnistMLP(**config)
    """
    def decorator(cls):
        registry = get_factory_registry()
        registry.register_component_creator(component_type, name, cls)
        cls._component_type = component_type
        cls._component_name = name
        return cls
    return decorator


# 装饰器用于自动注册工厂
def register_factory(factory_type: str):
    """
    工厂注册装饰器
    
    Args:
        factory_type: 工厂类型标识
        
    Example:
        @register_factory("model")
        class ModelFactory(FactoryInterface):
            pass
    """
    def decorator(cls):
        registry = get_factory_registry()
        registry.register_factory(factory_type, cls)
        cls._factory_type = factory_type
        return cls
    return decorator