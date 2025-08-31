"""
联邦学习组件抽象基类

定义了联邦学习框架中所有组件的通用接口和行为。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
from datetime import datetime
import logging
import uuid
from enum import Enum


class FederatedComponent(ABC):
    """
    联邦学习组件抽象基类
    
    所有联邦学习组件(客户端、服务器、聚合器等)的基类，
    提供通用的标识、配置、生命周期管理和日志功能。
    
    设计原则:
    - 单一职责: 每个组件只负责特定功能
    - 开闭原则: 通过继承扩展，不修改基类
    - 里氏替换: 子类可以完全替换基类使用
    """
    
    def __init__(self, 
                 component_id: Optional[str] = None,
                 config: Optional[Dict[str, Any]] = None,
                 logger: Optional[logging.Logger] = None):
        """
        初始化联邦学习组件
        
        Args:
            component_id: 组件唯一标识符，如果为None则自动生成
            config: 组件配置字典
            logger: 日志记录器，如果为None则创建默认日志器
        """
        self._component_id = component_id or str(uuid.uuid4())
        self._config = config or {}
        self._logger = logger or self._create_default_logger()
        self._created_at = datetime.now()
        self._status = ComponentStatus.CREATED
        self._metadata: Dict[str, Any] = {}
        
        # 验证配置
        self._validate_config()
        
        # 初始化组件
        self._initialize()
        
    @property
    def component_id(self) -> str:
        """获取组件ID"""
        return self._component_id
        
    @property
    def config(self) -> Dict[str, Any]:
        """获取组件配置"""
        return self._config.copy()
        
    @property
    def logger(self) -> logging.Logger:
        """获取日志记录器"""
        return self._logger
        
    @property
    def status(self) -> 'ComponentStatus':
        """获取组件状态"""
        return self._status
        
    @property
    def created_at(self) -> datetime:
        """获取创建时间"""
        return self._created_at
        
    @property
    def metadata(self) -> Dict[str, Any]:
        """获取组件元数据"""
        return self._metadata.copy()
        
    def get_config_value(self, key: str, default: Any = None) -> Any:
        """
        获取配置值，支持嵌套键访问
        
        Args:
            key: 配置键，支持点分隔的嵌套访问，如 'model.hidden_size'
            default: 默认值
            
        Returns:
            配置值或默认值
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
        
    def set_metadata(self, key: str, value: Any) -> None:
        """设置元数据"""
        self._metadata[key] = value
        
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """获取元数据"""
        return self._metadata.get(key, default)
        
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        更新组件配置
        
        Args:
            new_config: 新的配置字典
            
        Raises:
            ConfigValidationError: 当新配置无效时
        """
        old_config = self._config.copy()
        self._config.update(new_config)
        
        try:
            self._validate_config()
            self._on_config_updated(old_config, self._config)
        except Exception as e:
            # 恢复旧配置
            self._config = old_config
            raise e
            
    @abstractmethod
    def _validate_config(self) -> None:
        """
        验证组件配置
        
        子类必须实现此方法来验证特定的配置要求
        
        Raises:
            ConfigValidationError: 当配置无效时
        """
        pass
        
    @abstractmethod
    def _initialize(self) -> None:
        """
        初始化组件
        
        子类必须实现此方法来完成特定的初始化逻辑
        """
        pass
        
    def _create_default_logger(self) -> logging.Logger:
        """创建默认日志记录器"""
        logger = logging.getLogger(f"{self.__class__.__name__}_{self._component_id[:8]}")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '[%(asctime)s] %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
        
    def _set_status(self, status: 'ComponentStatus') -> None:
        """设置组件状态"""
        old_status = self._status
        self._status = status
        self._on_status_changed(old_status, status)
        
    def _on_config_updated(self, old_config: Dict[str, Any], new_config: Dict[str, Any]) -> None:
        """
        配置更新回调
        
        子类可以重写此方法来处理配置更新
        
        Args:
            old_config: 旧配置
            new_config: 新配置
        """
        self._logger.info(f"Configuration updated for component {self._component_id}")
        
    def _on_status_changed(self, old_status: 'ComponentStatus', new_status: 'ComponentStatus') -> None:
        """
        状态变更回调
        
        Args:
            old_status: 旧状态
            new_status: 新状态
        """
        self._logger.info(f"Status changed: {old_status.value} -> {new_status.value}")
        
    def __str__(self) -> str:
        """字符串表示"""
        return f"{self.__class__.__name__}(id={self._component_id[:8]}, status={self._status.value})"
        
    def __repr__(self) -> str:
        """详细字符串表示"""
        return (f"{self.__class__.__name__}("
                f"id='{self._component_id}', "
                f"status={self._status.value}, "
                f"created_at='{self._created_at.isoformat()}')")


class ComponentStatus(Enum):
    """组件状态枚举"""
    CREATED = "created"
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    STOPPED = "stopped"
    
    @classmethod
    def all_statuses(cls) -> list:
        """获取所有状态"""
        return [status for status in cls]


class ComponentRegistry:
    """
    组件注册表
    
    用于管理和追踪框架中的所有组件实例
    """
    
    _instance = None
    _components: Dict[str, FederatedComponent] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
        
    @classmethod
    def register(cls, component: FederatedComponent) -> None:
        """注册组件"""
        cls._components[component.component_id] = component
        
    @classmethod
    def unregister(cls, component_id: str) -> None:
        """取消注册组件"""
        cls._components.pop(component_id, None)
        
    @classmethod
    def get_component(cls, component_id: str) -> Optional[FederatedComponent]:
        """获取组件"""
        return cls._components.get(component_id)
        
    @classmethod
    def list_components(cls, component_type: Optional[type] = None) -> list:
        """列出所有组件"""
        if component_type is None:
            return list(cls._components.values())
        return [comp for comp in cls._components.values() 
                if isinstance(comp, component_type)]
        
    @classmethod
    def clear(cls) -> None:
        """清空注册表"""
        cls._components.clear()