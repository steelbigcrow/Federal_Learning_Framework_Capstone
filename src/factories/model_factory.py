"""
模型工厂实现

提供统一的模型创建和管理接口，支持：
- MNIST模型：MLP, Vision Transformer
- IMDB模型：RNN, LSTM, Transformer
- 模型参数验证和配置管理
- 动态模型注册和发现
"""

from typing import Dict, List, Any, Optional, Type
import logging
import torch.nn as nn

from ..core.interfaces.factory import ModelFactoryInterface, ComponentType
from ..core.exceptions import FactoryError
from ..models import create_model as legacy_create_model
from .factory_registry import register_factory, get_factory_registry


@register_factory("model")
class ModelFactory(ModelFactoryInterface):
    """
    模型工厂实现
    
    提供统一的模型创建接口，支持所有框架中的模型类型。
    继承现有的模型创建逻辑，同时添加工厂模式的功能。
    """
    
    def __init__(self):
        self._logger = logging.getLogger(__name__)
        self._registry = get_factory_registry()
        
        # 支持的数据集和对应的模型
        self._supported_models = {
            'mnist': ['mlp', 'vit'],
            'imdb': ['rnn', 'lstm', 'transformer']
        }
        
        # 模型配置验证规则
        self._config_validators = {
            'mnist': {
                'mlp': self._validate_mnist_mlp_config,
                'vit': self._validate_mnist_vit_config
            },
            'imdb': {
                'rnn': self._validate_imdb_rnn_config,
                'lstm': self._validate_imdb_lstm_config,
                'transformer': self._validate_imdb_transformer_config
            }
        }
        
        self._logger.info("ModelFactory initialized")
    
    def create(self, component_type: str, config: Dict[str, Any], **kwargs) -> nn.Module:
        """
        创建模型组件
        
        Args:
            component_type: 模型类型标识 (格式: "dataset_model", 如 "mnist_mlp")
            config: 模型配置
            **kwargs: 额外参数（如vocab_size）
            
        Returns:
            创建的模型实例
            
        Raises:
            FactoryError: 当创建失败时
        """
        try:
            # 解析组件类型
            if '_' not in component_type:
                raise FactoryError(f"Invalid model component type: {component_type}. Expected format: 'dataset_model'")
                
            dataset, model_type = component_type.split('_', 1)
            return self.create_model(dataset, model_type, config, **kwargs)
            
        except Exception as e:
            raise FactoryError(f"Failed to create model {component_type}: {e}")
    
    def register(self, component_type: str, creator_class: Type) -> None:
        """
        注册模型创建器
        
        Args:
            component_type: 模型类型标识
            creator_class: 创建器类
        """
        self._registry.register_component_creator(ComponentType.MODEL, component_type, creator_class)
        self._logger.info(f"Registered model creator: {component_type}")
    
    def unregister(self, component_type: str) -> None:
        """
        注销模型创建器
        
        Args:
            component_type: 模型类型标识
        """
        self._registry.unregister_component_creator(ComponentType.MODEL, component_type)
        self._logger.info(f"Unregistered model creator: {component_type}")
    
    def list_registered_types(self) -> List[str]:
        """
        列出所有已注册的模型类型
        
        Returns:
            已注册模型类型列表
        """
        builtin_types = []
        for dataset, models in self._supported_models.items():
            builtin_types.extend([f"{dataset}_{model}" for model in models])
            
        custom_types = self._registry.list_component_creators(ComponentType.MODEL)
        return builtin_types + custom_types
    
    def is_registered(self, component_type: str) -> bool:
        """
        检查模型类型是否已注册
        
        Args:
            component_type: 模型类型标识
            
        Returns:
            是否已注册
        """
        # 检查内置模型
        if '_' in component_type:
            dataset, model_type = component_type.split('_', 1)
            if dataset in self._supported_models:
                if model_type in self._supported_models[dataset]:
                    return True
        
        # 检查自定义注册的模型
        return self._registry.is_component_creator_registered(ComponentType.MODEL, component_type)
    
    def create_model(self, 
                     dataset: str, 
                     model_type: str, 
                     config: Dict[str, Any],
                     **kwargs) -> nn.Module:
        """
        创建模型实例
        
        Args:
            dataset: 数据集名称
            model_type: 模型类型
            config: 模型配置
            **kwargs: 额外参数（如vocab_size等）
            
        Returns:
            创建的模型实例
            
        Raises:
            FactoryError: 当创建失败时
        """
        try:
            # 验证数据集和模型类型
            dataset = dataset.lower()
            model_type = model_type.lower()
            
            if dataset not in self._supported_models:
                raise FactoryError(f"Unsupported dataset: {dataset}")
                
            if model_type not in self._supported_models[dataset]:
                raise FactoryError(f"Unsupported model '{model_type}' for dataset '{dataset}'")
            
            # 验证配置
            self._validate_config(dataset, model_type, config, kwargs)
            
            # 使用现有的模型创建逻辑
            model = legacy_create_model(dataset, model_type, config, kwargs)
            
            self._logger.info(f"Created model: {dataset}_{model_type}")
            return model
            
        except Exception as e:
            if isinstance(e, FactoryError):
                raise
            raise FactoryError(f"Failed to create model {dataset}_{model_type}: {e}")
    
    def get_supported_datasets(self) -> List[str]:
        """获取支持的数据集列表"""
        return list(self._supported_models.keys())
    
    def get_supported_models(self, dataset: str) -> List[str]:
        """获取指定数据集支持的模型列表"""
        return self._supported_models.get(dataset.lower(), [])
    
    def _validate_config(self, 
                        dataset: str, 
                        model_type: str, 
                        config: Dict[str, Any], 
                        extra: Dict[str, Any]) -> None:
        """
        验证模型配置
        
        Args:
            dataset: 数据集名称
            model_type: 模型类型
            config: 模型配置
            extra: 额外参数
            
        Raises:
            FactoryError: 当配置无效时
        """
        if dataset in self._config_validators:
            if model_type in self._config_validators[dataset]:
                validator = self._config_validators[dataset][model_type]
                validator(config, extra)
    
    def _validate_mnist_mlp_config(self, config: Dict[str, Any], extra: Dict[str, Any]) -> None:
        """验证MNIST MLP配置"""
        required_fields = ['input_size', 'hidden_sizes', 'num_classes']
        for field in required_fields:
            if field not in config:
                self._logger.warning(f"Missing {field} in MLP config, using default")
        
        if 'hidden_sizes' in config and not isinstance(config['hidden_sizes'], list):
            raise FactoryError("hidden_sizes must be a list")
    
    def _validate_mnist_vit_config(self, config: Dict[str, Any], extra: Dict[str, Any]) -> None:
        """验证MNIST Vision Transformer配置"""
        if 'patch_size' in config and 'image_size' in config:
            if config['image_size'] % config['patch_size'] != 0:
                raise FactoryError("image_size must be divisible by patch_size")
        
        if 'nhead' in config and 'emb_dim' in config:
            if config['emb_dim'] % config['nhead'] != 0:
                raise FactoryError("emb_dim must be divisible by nhead")
    
    def _validate_imdb_rnn_config(self, config: Dict[str, Any], extra: Dict[str, Any]) -> None:
        """验证IMDB RNN配置"""
        if 'vocab_size' not in extra:
            raise FactoryError("IMDB models require vocab_size in extra parameters")
        
        if extra['vocab_size'] <= 0:
            raise FactoryError("vocab_size must be positive")
    
    def _validate_imdb_lstm_config(self, config: Dict[str, Any], extra: Dict[str, Any]) -> None:
        """验证IMDB LSTM配置"""
        self._validate_imdb_rnn_config(config, extra)  # 使用相同的验证逻辑
    
    def _validate_imdb_transformer_config(self, config: Dict[str, Any], extra: Dict[str, Any]) -> None:
        """验证IMDB Transformer配置"""
        self._validate_imdb_rnn_config(config, extra)
        
        if 'nhead' in config and 'embedding_dim' in config:
            if config['embedding_dim'] % config['nhead'] != 0:
                raise FactoryError("embedding_dim must be divisible by nhead")
    
    def get_model_info(self, dataset: str, model_type: str) -> Dict[str, Any]:
        """
        获取模型信息
        
        Args:
            dataset: 数据集名称
            model_type: 模型类型
            
        Returns:
            包含模型信息的字典
        """
        dataset = dataset.lower()
        model_type = model_type.lower()
        
        if dataset not in self._supported_models or model_type not in self._supported_models[dataset]:
            raise FactoryError(f"Unknown model: {dataset}_{model_type}")
        
        info = {
            'dataset': dataset,
            'model_type': model_type,
            'identifier': f"{dataset}_{model_type}",
            'supported': True
        }
        
        # 添加模型特定信息
        if dataset == 'mnist':
            if model_type == 'mlp':
                info.update({
                    'description': 'Multi-Layer Perceptron for MNIST digit classification',
                    'input_shape': (784,),
                    'output_classes': 10,
                    'default_config': {
                        'input_size': 784,
                        'hidden_sizes': [512, 256],
                        'num_classes': 10
                    }
                })
            elif model_type == 'vit':
                info.update({
                    'description': 'Vision Transformer for MNIST digit classification',
                    'input_shape': (1, 28, 28),
                    'output_classes': 10,
                    'default_config': {
                        'image_size': 28,
                        'patch_size': 7,
                        'emb_dim': 128,
                        'depth': 4,
                        'nhead': 4,
                        'num_classes': 10
                    }
                })
        elif dataset == 'imdb':
            base_info = {
                'input_type': 'text',
                'output_classes': 2,
                'requires_vocab_size': True
            }
            
            if model_type == 'rnn':
                info.update(base_info)
                info.update({
                    'description': 'Recurrent Neural Network for IMDB sentiment classification',
                    'default_config': {
                        'embedding_dim': 128,
                        'hidden_size': 128,
                        'num_layers': 1,
                        'bidirectional': False
                    }
                })
            elif model_type == 'lstm':
                info.update(base_info)
                info.update({
                    'description': 'Long Short-Term Memory for IMDB sentiment classification',
                    'default_config': {
                        'embedding_dim': 128,
                        'hidden_size': 128,
                        'num_layers': 1,
                        'bidirectional': False
                    }
                })
            elif model_type == 'transformer':
                info.update(base_info)
                info.update({
                    'description': 'Transformer model for IMDB sentiment classification',
                    'default_config': {
                        'embedding_dim': 128,
                        'nhead': 4,
                        'num_layers': 2,
                        'hidden_dim': 256,
                        'max_seq_len': 256
                    }
                })
        
        return info
    
    def create_model_with_defaults(self, 
                                  dataset: str, 
                                  model_type: str,
                                  config_overrides: Optional[Dict[str, Any]] = None,
                                  **kwargs) -> nn.Module:
        """
        使用默认配置创建模型
        
        Args:
            dataset: 数据集名称
            model_type: 模型类型
            config_overrides: 配置覆盖
            **kwargs: 额外参数
            
        Returns:
            创建的模型实例
        """
        model_info = self.get_model_info(dataset, model_type)
        default_config = model_info.get('default_config', {})
        
        if config_overrides:
            default_config.update(config_overrides)
        
        return self.create_model(dataset, model_type, default_config, **kwargs)