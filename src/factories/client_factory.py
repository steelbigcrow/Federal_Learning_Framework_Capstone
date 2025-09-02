"""
客户端工厂实现

提供统一的联邦学习客户端创建和管理接口，支持：
- 标准客户端创建
- 批量客户端创建
- 客户端配置验证和管理
- 与现有客户端实现的集成
"""

from typing import Dict, List, Any, Optional, Type, Callable
import logging
import torch
from torch.utils.data import DataLoader

from ..core.interfaces.factory import ClientFactoryInterface, ComponentType
from ..core.base.client import AbstractClient
from ..core.exceptions import FactoryError
from ..implementations.clients.federated_client import FederatedClient as LegacyClient
from .factory_registry import register_factory, get_factory_registry


@register_factory("client")
class ClientFactory(ClientFactoryInterface):
    """
    客户端工厂实现
    
    提供统一的联邦学习客户端创建接口，支持现有的Client实现和未来的扩展。
    """
    
    def __init__(self):
        self._logger = logging.getLogger(__name__)
        self._registry = get_factory_registry()
        
        # 支持的客户端类型
        self._supported_types = ['standard', 'legacy']
        
        # 客户端创建器映射
        self._client_creators = {
            'standard': self._create_standard_client,
            'legacy': self._create_legacy_client
        }
        
        # 默认配置
        self._default_config = {
            'client_type': 'legacy',  # 默认使用现有的实现
            'optimizer': {
                'name': 'adam',
                'lr': 1e-3,
                'weight_decay': 0.0
            },
            'use_amp': False,
            'test_ratio': 0.3,
            'device': 'cpu'
        }
        
        self._logger.info("ClientFactory initialized")
    
    def create(self, component_type: str, config: Dict[str, Any], **kwargs) -> Any:
        """
        创建客户端组件
        
        Args:
            component_type: 客户端类型标识
            config: 客户端配置
            **kwargs: 额外参数
            
        Returns:
            创建的客户端实例
            
        Raises:
            FactoryError: 当创建失败时
        """
        try:
            if component_type not in self._supported_types:
                raise FactoryError(f"Unsupported client type: {component_type}")
            
            # 提取必要的参数
            client_id = kwargs.get('client_id', config.get('client_id'))
            train_dataloader = kwargs.get('train_dataloader', config.get('train_dataloader'))
            
            if client_id is None:
                raise FactoryError("client_id is required")
            if train_dataloader is None:
                raise FactoryError("train_dataloader is required")
                
            return self.create_client(client_id, train_dataloader, config)
            
        except Exception as e:
            raise FactoryError(f"Failed to create client {component_type}: {e}")
    
    def register(self, component_type: str, creator_class: Type) -> None:
        """
        注册客户端创建器
        
        Args:
            component_type: 客户端类型标识
            creator_class: 创建器类
        """
        self._registry.register_component_creator(ComponentType.CLIENT, component_type, creator_class)
        self._logger.info(f"Registered client creator: {component_type}")
    
    def unregister(self, component_type: str) -> None:
        """
        注销客户端创建器
        
        Args:
            component_type: 客户端类型标识
        """
        self._registry.unregister_component_creator(ComponentType.CLIENT, component_type)
        self._logger.info(f"Unregistered client creator: {component_type}")
    
    def list_registered_types(self) -> List[str]:
        """
        列出所有已注册的客户端类型
        
        Returns:
            已注册客户端类型列表
        """
        builtin_types = list(self._supported_types)
        custom_types = self._registry.list_component_creators(ComponentType.CLIENT)
        return builtin_types + custom_types
    
    def is_registered(self, component_type: str) -> bool:
        """
        检查客户端类型是否已注册
        
        Args:
            component_type: 客户端类型标识
            
        Returns:
            是否已注册
        """
        if component_type in self._supported_types:
            return True
        return self._registry.is_component_creator_registered(ComponentType.CLIENT, component_type)
    
    def create_client(self, 
                      client_id: int,
                      train_dataloader: DataLoader,
                      config: Dict[str, Any]) -> Any:
        """
        创建客户端实例
        
        Args:
            client_id: 客户端ID
            train_dataloader: 训练数据加载器
            config: 客户端配置
            
        Returns:
            创建的客户端实例
            
        Raises:
            FactoryError: 当创建失败时
        """
        try:
            # 合并默认配置
            full_config = {**self._default_config, **config}
            
            # 验证配置
            self._validate_config(full_config)
            
            # 获取客户端类型
            client_type = full_config.get('client_type', 'legacy')
            
            if client_type not in self._client_creators:
                raise FactoryError(f"Unknown client type: {client_type}")
            
            # 创建客户端
            creator = self._client_creators[client_type]
            client = creator(client_id, train_dataloader, full_config)
            
            self._logger.info(f"Created client {client_id} of type {client_type}")
            return client
            
        except Exception as e:
            if isinstance(e, FactoryError):
                raise
            raise FactoryError(f"Failed to create client {client_id}: {e}")
    
    def create_clients_batch(self, 
                            dataloaders: List[DataLoader],
                            config: Dict[str, Any]) -> List[Any]:
        """
        批量创建客户端
        
        Args:
            dataloaders: 数据加载器列表
            config: 客户端配置
            
        Returns:
            创建的客户端列表
            
        Raises:
            FactoryError: 当创建失败时
        """
        try:
            clients = []
            
            for i, dataloader in enumerate(dataloaders):
                try:
                    client = self.create_client(i, dataloader, config)
                    clients.append(client)
                except Exception as e:
                    raise FactoryError(f"Failed to create client {i}: {e}")
            
            self._logger.info(f"Created batch of {len(clients)} clients")
            return clients
            
        except Exception as e:
            if isinstance(e, FactoryError):
                raise
            raise FactoryError(f"Failed to create clients batch: {e}")
    
    def _create_legacy_client(self, 
                             client_id: int,
                             train_dataloader: DataLoader,
                             config: Dict[str, Any]) -> LegacyClient:
        """创建标准（遗留）客户端"""
        # 提取必要的参数
        model_ctor = config.get('model_constructor')
        if model_ctor is None:
            raise FactoryError("model_constructor is required for legacy client")
            
        device = config.get('device', 'cpu')
        optimizer_cfg = config.get('optimizer', {})
        use_amp = config.get('use_amp', False)
        test_ratio = config.get('test_ratio', 0.3)
        
        # 创建客户端配置
        client_config = {
            'optimizer': optimizer_cfg,
            'use_amp': use_amp,
            'test_ratio': test_ratio
        }
        
        # 创建客户端
        client = LegacyClient(
            client_id=client_id,
            model_ctor=model_ctor,
            train_data_loader=train_dataloader,
            config=client_config,
            device=device
        )
        
        return client
    
    def _create_standard_client(self, 
                               client_id: int,
                               train_dataloader: DataLoader,
                               config: Dict[str, Any]) -> AbstractClient:
        """创建基于抽象基类的标准客户端（预留给未来实现）"""
        raise FactoryError("Standard client implementation not available yet")
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """
        验证客户端配置
        
        Args:
            config: 客户端配置
            
        Raises:
            FactoryError: 当配置无效时
        """
        required_fields = ['client_type']
        for field in required_fields:
            if field not in config:
                raise FactoryError(f"Missing required field: {field}")
        
        # 验证客户端类型
        client_type = config['client_type']
        if client_type not in self._supported_types:
            raise FactoryError(f"Unsupported client type: {client_type}")
        
        # 验证优化器配置
        if 'optimizer' in config:
            optimizer_config = config['optimizer']
            if not isinstance(optimizer_config, dict):
                raise FactoryError("optimizer config must be a dictionary")
                
            if 'lr' in optimizer_config:
                lr = optimizer_config['lr']
                if not isinstance(lr, (int, float)) or lr <= 0:
                    raise FactoryError("learning rate must be a positive number")
        
        # 验证其他参数
        if 'test_ratio' in config:
            test_ratio = config['test_ratio']
            if not isinstance(test_ratio, (int, float)) or not (0 < test_ratio < 1):
                raise FactoryError("test_ratio must be a number between 0 and 1")
        
        if 'use_amp' in config:
            if not isinstance(config['use_amp'], bool):
                raise FactoryError("use_amp must be a boolean")
    
    def get_client_info(self, client_type: str) -> Dict[str, Any]:
        """
        获取客户端类型信息
        
        Args:
            client_type: 客户端类型
            
        Returns:
            包含客户端信息的字典
        """
        if client_type not in self._supported_types:
            raise FactoryError(f"Unknown client type: {client_type}")
        
        info = {
            'type': client_type,
            'supported': True
        }
        
        if client_type == 'legacy':
            info.update({
                'description': 'Standard federated learning client implementation',
                'features': [
                    'Local training and evaluation',
                    'Automatic train/test split',
                    'Adam/AdamW optimizer support',
                    'Mixed precision training',
                    'Epoch-by-epoch metrics tracking'
                ],
                'required_config': [
                    'model_constructor',
                    'device'
                ],
                'default_config': self._default_config
            })
        elif client_type == 'standard':
            info.update({
                'description': 'Abstract base class based client (future implementation)',
                'features': [
                    'OOP-based design',
                    'Plugin system integration',
                    'Strategy pattern support'
                ],
                'status': 'not_implemented'
            })
        
        return info
    
    def create_client_with_defaults(self,
                                   client_id: int,
                                   train_dataloader: DataLoader,
                                   model_constructor: Callable,
                                   device: str = 'cpu',
                                   config_overrides: Optional[Dict[str, Any]] = None) -> Any:
        """
        使用默认配置创建客户端
        
        Args:
            client_id: 客户端ID
            train_dataloader: 训练数据加载器
            model_constructor: 模型构造函数
            device: 计算设备
            config_overrides: 配置覆盖
            
        Returns:
            创建的客户端实例
        """
        config = {
            'model_constructor': model_constructor,
            'device': device
        }
        
        if config_overrides:
            config.update(config_overrides)
        
        return self.create_client(client_id, train_dataloader, config)
    
    def validate_dataloaders(self, dataloaders: List[DataLoader]) -> bool:
        """
        验证数据加载器列表
        
        Args:
            dataloaders: 数据加载器列表
            
        Returns:
            验证是否通过
            
        Raises:
            FactoryError: 当验证失败时
        """
        if not isinstance(dataloaders, list):
            raise FactoryError("dataloaders must be a list")
        
        if len(dataloaders) == 0:
            raise FactoryError("dataloaders cannot be empty")
        
        for i, dataloader in enumerate(dataloaders):
            if not isinstance(dataloader, DataLoader):
                raise FactoryError(f"dataloaders[{i}] must be a DataLoader instance")
            
            if len(dataloader.dataset) == 0:
                raise FactoryError(f"dataloaders[{i}] has empty dataset")
        
        self._logger.debug(f"Validated {len(dataloaders)} dataloaders")
        return True
    
    def get_factory_stats(self) -> Dict[str, Any]:
        """
        获取工厂统计信息
        
        Returns:
            包含统计信息的字典
        """
        return {
            'supported_types': len(self._supported_types),
            'available_types': self._supported_types,
            'default_config': self._default_config,
            'custom_creators': len(self._registry.list_component_creators(ComponentType.CLIENT))
        }