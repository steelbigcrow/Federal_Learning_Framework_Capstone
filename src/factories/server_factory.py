"""
服务器工厂实现

提供统一的联邦学习服务器创建和管理接口，支持：
- 标准服务器创建
- 服务器配置验证和管理
- 与现有服务器实现的集成
- 路径管理和日志配置
"""

from typing import Dict, List, Any, Optional, Type, Callable
import logging

from ..core.interfaces.factory import ServerFactoryInterface, ComponentType
from ..core.base.server import AbstractServer
from ..core.exceptions import FactoryError
from ..implementations.servers.federated_server import FederatedServer as LegacyServer
from ..utils.paths import PathManager
from .factory_registry import register_factory, get_factory_registry


@register_factory("server")
class ServerFactory(ServerFactoryInterface):
    """
    服务器工厂实现
    
    提供统一的联邦学习服务器创建接口，支持现有的Server实现和未来的扩展。
    """
    
    def __init__(self):
        self._logger = logging.getLogger(__name__)
        self._registry = get_factory_registry()
        
        # 支持的服务器类型
        self._supported_types = ['standard', 'legacy']
        
        # 服务器创建器映射
        self._server_creators = {
            'standard': self._create_standard_server,
            'legacy': self._create_legacy_server
        }
        
        # 默认配置
        self._default_config = {
            'server_type': 'legacy',  # 默认使用现有的实现
            'device': 'cpu',
            'save_client_each_round': True,
            'lora_cfg': {},
            'adalora_cfg': {},
            'model_info': {}
        }
        
        self._logger.info("ServerFactory initialized")
    
    def create(self, component_type: str, config: Dict[str, Any], **kwargs) -> Any:
        """
        创建服务器组件
        
        Args:
            component_type: 服务器类型标识
            config: 服务器配置
            **kwargs: 额外参数
            
        Returns:
            创建的服务器实例
            
        Raises:
            FactoryError: 当创建失败时
        """
        try:
            if component_type not in self._supported_types:
                raise FactoryError(f"Unsupported server type: {component_type}")
            
            # 提取必要的参数
            model_constructor = kwargs.get('model_constructor', config.get('model_constructor'))
            clients = kwargs.get('clients', config.get('clients'))
            
            if model_constructor is None:
                raise FactoryError("model_constructor is required")
            if clients is None:
                raise FactoryError("clients list is required")
                
            return self.create_server(model_constructor, clients, config)
            
        except Exception as e:
            raise FactoryError(f"Failed to create server {component_type}: {e}")
    
    def register(self, component_type: str, creator_class: Type) -> None:
        """
        注册服务器创建器
        
        Args:
            component_type: 服务器类型标识
            creator_class: 创建器类
        """
        self._registry.register_component_creator(ComponentType.SERVER, component_type, creator_class)
        self._logger.info(f"Registered server creator: {component_type}")
    
    def unregister(self, component_type: str) -> None:
        """
        注销服务器创建器
        
        Args:
            component_type: 服务器类型标识
        """
        self._registry.unregister_component_creator(ComponentType.SERVER, component_type)
        self._logger.info(f"Unregistered server creator: {component_type}")
    
    def list_registered_types(self) -> List[str]:
        """
        列出所有已注册的服务器类型
        
        Returns:
            已注册服务器类型列表
        """
        builtin_types = list(self._supported_types)
        custom_types = self._registry.list_component_creators(ComponentType.SERVER)
        return builtin_types + custom_types
    
    def is_registered(self, component_type: str) -> bool:
        """
        检查服务器类型是否已注册
        
        Args:
            component_type: 服务器类型标识
            
        Returns:
            是否已注册
        """
        if component_type in self._supported_types:
            return True
        return self._registry.is_component_creator_registered(ComponentType.SERVER, component_type)
    
    def create_server(self, 
                      model_constructor: Callable,
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
            
        Raises:
            FactoryError: 当创建失败时
        """
        try:
            # 合并默认配置
            full_config = {**self._default_config, **config}
            
            # 验证配置
            self._validate_config(full_config, clients)
            
            # 获取服务器类型
            server_type = full_config.get('server_type', 'legacy')
            
            if server_type not in self._server_creators:
                raise FactoryError(f"Unknown server type: {server_type}")
            
            # 创建服务器
            creator = self._server_creators[server_type]
            server = creator(model_constructor, clients, full_config)
            
            self._logger.info(f"Created server of type {server_type} with {len(clients)} clients")
            return server
            
        except Exception as e:
            if isinstance(e, FactoryError):
                raise
            raise FactoryError(f"Failed to create server: {e}")
    
    def _create_legacy_server(self, 
                             model_constructor: Callable,
                             clients: List[Any],
                             config: Dict[str, Any]) -> LegacyServer:
        """创建标准（遗留）服务器"""
        # 创建路径管理器
        path_manager = self._create_path_manager(config)
        
        # 提取配置参数
        device = config.get('device', 'cpu')
        lora_cfg = config.get('lora_cfg', {})
        adalora_cfg = config.get('adalora_cfg', {})
        save_client_each_round = config.get('save_client_each_round', True)
        model_info = config.get('model_info', {})
        
        # 创建服务器配置
        server_config = {
            'federated': {
                'num_rounds': 10,  # 默认值
                'local_epochs': 5  # 默认值
            },
            'lora_cfg': lora_cfg,
            'adalora_cfg': adalora_cfg,
            'save_client_each_round': save_client_each_round,
            'model_info': model_info
        }
        
        # 创建服务器
        server = LegacyServer(
            model_constructor=model_constructor,
            clients=clients,
            path_manager=path_manager,
            config=server_config,
            device=device
        )
        
        return server
    
    def _create_standard_server(self, 
                               model_constructor: Callable,
                               clients: List[Any],
                               config: Dict[str, Any]) -> AbstractServer:
        """创建基于抽象基类的标准服务器（预留给未来实现）"""
        raise FactoryError("Standard server implementation not available yet")
    
    def _create_path_manager(self, config: Dict[str, Any]) -> PathManager:
        """
        创建路径管理器
        
        Args:
            config: 服务器配置
            
        Returns:
            PathManager实例
            
        Raises:
            FactoryError: 当创建失败时
        """
        # 从配置中获取路径信息
        base_dir = config.get('base_dir', 'outputs')
        run_name = config.get('run_name')
        dataset = config.get('dataset', 'unknown')
        model_type = config.get('model_type', 'unknown')
        
        if run_name is None:
            raise FactoryError("run_name is required for path management")
        
        try:
            # 根据是否使用LoRA或AdaLoRA决定输出目录
            lora_cfg = config.get('lora_cfg', {})
            adalora_cfg = config.get('adalora_cfg', {})
            
            if adalora_cfg and adalora_cfg.get('replaced_modules'):
                # AdaLoRA模式
                output_subdir = 'adaloras'
                full_run_name = f"{dataset}_{model_type}_adalora_{run_name}"
            elif lora_cfg and lora_cfg.get('replaced_modules'):
                # LoRA模式
                output_subdir = 'loras'
                full_run_name = f"{dataset}_{model_type}_lora_{run_name}"
            else:
                # 标准模式
                output_subdir = 'models'
                full_run_name = f"{dataset}_{model_type}_{run_name}"
            
            path_manager = PathManager(
                base_dir=base_dir,
                output_subdir=output_subdir,
                run_name=full_run_name
            )
            
            self._logger.debug(f"Created path manager: {output_subdir}/{full_run_name}")
            return path_manager
            
        except Exception as e:
            raise FactoryError(f"Failed to create path manager: {e}")
    
    def _validate_config(self, config: Dict[str, Any], clients: List[Any]) -> None:
        """
        验证服务器配置
        
        Args:
            config: 服务器配置
            clients: 客户端列表
            
        Raises:
            FactoryError: 当配置无效时
        """
        required_fields = ['server_type', 'run_name']
        for field in required_fields:
            if field not in config:
                raise FactoryError(f"Missing required field: {field}")
        
        # 验证服务器类型
        server_type = config['server_type']
        if server_type not in self._supported_types:
            raise FactoryError(f"Unsupported server type: {server_type}")
        
        # 验证客户端列表
        if not isinstance(clients, list):
            raise FactoryError("clients must be a list")
        
        if len(clients) == 0:
            raise FactoryError("clients list cannot be empty")
        
        # 验证run_name
        run_name = config['run_name']
        if not isinstance(run_name, str) or not run_name.strip():
            raise FactoryError("run_name must be a non-empty string")
        
        # 验证设备
        device = config.get('device', 'cpu')
        if not isinstance(device, str):
            raise FactoryError("device must be a string")
        
        # 验证LoRA配置
        if 'lora_cfg' in config:
            lora_cfg = config['lora_cfg']
            if not isinstance(lora_cfg, dict):
                raise FactoryError("lora_cfg must be a dictionary")
        
        # 验证AdaLoRA配置
        if 'adalora_cfg' in config:
            adalora_cfg = config['adalora_cfg']
            if not isinstance(adalora_cfg, dict):
                raise FactoryError("adalora_cfg must be a dictionary")
        
        # 验证模型信息
        if 'model_info' in config:
            model_info = config['model_info']
            if not isinstance(model_info, dict):
                raise FactoryError("model_info must be a dictionary")
    
    def get_server_info(self, server_type: str) -> Dict[str, Any]:
        """
        获取服务器类型信息
        
        Args:
            server_type: 服务器类型
            
        Returns:
            包含服务器信息的字典
        """
        if server_type not in self._supported_types:
            raise FactoryError(f"Unknown server type: {server_type}")
        
        info = {
            'type': server_type,
            'supported': True
        }
        
        if server_type == 'legacy':
            info.update({
                'description': 'Standard federated learning server implementation',
                'features': [
                    'Model aggregation (FedAvg, LoRA, AdaLoRA)',
                    'Client coordination and management',
                    'Checkpoint saving and loading',
                    'Metrics collection and logging',
                    'Visualization and plotting',
                    'Path management for outputs'
                ],
                'required_config': [
                    'run_name',
                    'dataset',
                    'model_type'
                ],
                'optional_config': [
                    'device',
                    'lora_cfg',
                    'adalora_cfg',
                    'save_client_each_round',
                    'model_info',
                    'base_dir'
                ],
                'default_config': self._default_config
            })
        elif server_type == 'standard':
            info.update({
                'description': 'Abstract base class based server (future implementation)',
                'features': [
                    'OOP-based design',
                    'Plugin system integration',
                    'Strategy pattern support',
                    'Event-driven architecture'
                ],
                'status': 'not_implemented'
            })
        
        return info
    
    def create_server_with_defaults(self,
                                   model_constructor: Callable,
                                   clients: List[Any],
                                   run_name: str,
                                   dataset: str,
                                   model_type: str,
                                   device: str = 'cpu',
                                   config_overrides: Optional[Dict[str, Any]] = None) -> Any:
        """
        使用默认配置创建服务器
        
        Args:
            model_constructor: 模型构造函数
            clients: 客户端列表
            run_name: 运行名称
            dataset: 数据集名称
            model_type: 模型类型
            device: 计算设备
            config_overrides: 配置覆盖
            
        Returns:
            创建的服务器实例
        """
        config = {
            'run_name': run_name,
            'dataset': dataset,
            'model_type': model_type,
            'device': device,
            'model_info': {
                'dataset': dataset,
                'model_type': model_type
            }
        }
        
        if config_overrides:
            config.update(config_overrides)
        
        return self.create_server(model_constructor, clients, config)
    
    def validate_clients(self, clients: List[Any]) -> bool:
        """
        验证客户端列表
        
        Args:
            clients: 客户端列表
            
        Returns:
            验证是否通过
            
        Raises:
            FactoryError: 当验证失败时
        """
        if not isinstance(clients, list):
            raise FactoryError("clients must be a list")
        
        if len(clients) == 0:
            raise FactoryError("clients cannot be empty")
        
        # 检查客户端是否有必要的方法
        required_methods = ['local_train_and_eval']
        for i, client in enumerate(clients):
            for method_name in required_methods:
                if not hasattr(client, method_name):
                    raise FactoryError(f"client[{i}] missing required method: {method_name}")
            
            # 检查客户端是否有ID
            if not hasattr(client, 'id'):
                raise FactoryError(f"client[{i}] missing 'id' attribute")
        
        # 检查客户端ID是否唯一
        client_ids = [client.id for client in clients]
        if len(client_ids) != len(set(client_ids)):
            raise FactoryError("client IDs must be unique")
        
        self._logger.debug(f"Validated {len(clients)} clients")
        return True
    
    def get_supported_training_modes(self) -> Dict[str, Dict[str, Any]]:
        """
        获取支持的训练模式信息
        
        Returns:
            训练模式信息字典
        """
        return {
            'standard': {
                'name': 'Standard Federated Learning',
                'description': 'Traditional FedAvg with full model training',
                'aggregation': 'fedavg',
                'trainable_params': 'all',
                'config_key': None
            },
            'lora': {
                'name': 'LoRA Fine-tuning',
                'description': 'Low-Rank Adaptation for parameter-efficient fine-tuning',
                'aggregation': 'lora_fedavg',
                'trainable_params': 'lora_weights + classifier',
                'config_key': 'lora_cfg'
            },
            'adalora': {
                'name': 'AdaLoRA Fine-tuning',
                'description': 'Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning',
                'aggregation': 'adalora_fedavg',
                'trainable_params': 'adalora_weights + classifier',
                'config_key': 'adalora_cfg'
            }
        }
    
    def detect_training_mode(self, config: Dict[str, Any]) -> str:
        """
        检测训练模式
        
        Args:
            config: 服务器配置
            
        Returns:
            训练模式名称
        """
        adalora_cfg = config.get('adalora_cfg', {})
        lora_cfg = config.get('lora_cfg', {})
        
        if adalora_cfg and adalora_cfg.get('replaced_modules'):
            return 'adalora'
        elif lora_cfg and lora_cfg.get('replaced_modules'):
            return 'lora'
        else:
            return 'standard'
    
    def get_factory_stats(self) -> Dict[str, Any]:
        """
        获取工厂统计信息
        
        Returns:
            包含统计信息的字典
        """
        return {
            'supported_types': len(self._supported_types),
            'available_types': self._supported_types,
            'training_modes': list(self.get_supported_training_modes().keys()),
            'default_config': self._default_config,
            'custom_creators': len(self._registry.list_component_creators(ComponentType.SERVER))
        }