"""
数据集工厂实现

提供统一的数据集创建和管理接口，支持：
- MNIST数据集：图像分类数据
- IMDB数据集：文本情感分析数据
- 数据集分割：标签偏移（label shift）分割策略
- 数据加载器创建和配置管理
"""

from typing import Dict, List, Any, Optional, Type, Tuple, Union
import logging
import torch
from torch.utils.data import DataLoader, Dataset

from ..core.interfaces.factory import DatasetFactoryInterface, ComponentType
from ..core.exceptions import FactoryError
from ..datasets import (
    get_mnist_datasets, get_imdb_splits,
    partition_mnist_label_shift, partition_imdb_label_shift,
    CollateText
)
from .factory_registry import register_factory, get_factory_registry


@register_factory("dataset")
class DatasetFactory(DatasetFactoryInterface):
    """
    数据集工厂实现
    
    提供统一的数据集和数据加载器创建接口，支持联邦学习场景下的数据分割。
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        self._logger = logging.getLogger(__name__)
        self._registry = get_factory_registry()
        self._cache_dir = cache_dir
        
        # 支持的数据集
        self._supported_datasets = ['mnist', 'imdb']
        
        # 支持的分割策略
        self._partition_strategies = {
            'mnist': ['label_shift'],
            'imdb': ['label_shift']
        }
        
        # 数据集创建函数映射
        self._dataset_creators = {
            'mnist': self._create_mnist_dataset,
            'imdb': self._create_imdb_dataset
        }
        
        # 分割函数映射
        self._partition_functions = {
            'mnist': {
                'label_shift': partition_mnist_label_shift
            },
            'imdb': {
                'label_shift': partition_imdb_label_shift
            }
        }
        
        self._logger.info(f"DatasetFactory initialized with cache_dir: {cache_dir}")
    
    def create(self, component_type: str, config: Dict[str, Any], **kwargs) -> Any:
        """
        创建数据集组件
        
        Args:
            component_type: 数据集类型标识
            config: 数据集配置
            **kwargs: 额外参数
            
        Returns:
            创建的数据集实例
            
        Raises:
            FactoryError: 当创建失败时
        """
        try:
            if component_type not in self._supported_datasets:
                raise FactoryError(f"Unsupported dataset type: {component_type}")
            
            split = config.get('split', 'train')
            return self.create_dataset(component_type, split, config)
            
        except Exception as e:
            raise FactoryError(f"Failed to create dataset {component_type}: {e}")
    
    def register(self, component_type: str, creator_class: Type) -> None:
        """
        注册数据集创建器
        
        Args:
            component_type: 数据集类型标识
            creator_class: 创建器类
        """
        self._registry.register_component_creator(ComponentType.DATASET, component_type, creator_class)
        self._logger.info(f"Registered dataset creator: {component_type}")
    
    def unregister(self, component_type: str) -> None:
        """
        注销数据集创建器
        
        Args:
            component_type: 数据集类型标识
        """
        self._registry.unregister_component_creator(ComponentType.DATASET, component_type)
        self._logger.info(f"Unregistered dataset creator: {component_type}")
    
    def list_registered_types(self) -> List[str]:
        """
        列出所有已注册的数据集类型
        
        Returns:
            已注册数据集类型列表
        """
        builtin_types = list(self._supported_datasets)
        custom_types = self._registry.list_component_creators(ComponentType.DATASET)
        return builtin_types + custom_types
    
    def is_registered(self, component_type: str) -> bool:
        """
        检查数据集类型是否已注册
        
        Args:
            component_type: 数据集类型标识
            
        Returns:
            是否已注册
        """
        if component_type in self._supported_datasets:
            return True
        return self._registry.is_component_creator_registered(ComponentType.DATASET, component_type)
    
    def create_dataset(self, 
                       dataset_name: str, 
                       split: str,
                       config: Dict[str, Any]) -> Dataset:
        """
        创建数据集实例
        
        Args:
            dataset_name: 数据集名称
            split: 数据分割类型（train/test/val）
            config: 数据集配置
            
        Returns:
            创建的数据集实例
            
        Raises:
            FactoryError: 当创建失败时
        """
        try:
            dataset_name = dataset_name.lower()
            split = split.lower()
            
            if dataset_name not in self._supported_datasets:
                raise FactoryError(f"Unsupported dataset: {dataset_name}")
            
            if split not in ['train', 'test', 'val']:
                raise FactoryError(f"Invalid split: {split}. Must be one of: train, test, val")
            
            creator = self._dataset_creators[dataset_name]
            dataset = creator(split, config)
            
            self._logger.info(f"Created dataset: {dataset_name} ({split})")
            return dataset
            
        except Exception as e:
            if isinstance(e, FactoryError):
                raise
            raise FactoryError(f"Failed to create dataset {dataset_name} ({split}): {e}")
    
    def create_dataloader(self, 
                          dataset: Dataset, 
                          config: Dict[str, Any]) -> DataLoader:
        """
        创建数据加载器
        
        Args:
            dataset: 数据集实例
            config: 数据加载器配置
            
        Returns:
            创建的数据加载器实例
            
        Raises:
            FactoryError: 当创建失败时
        """
        try:
            # 默认配置
            default_config = {
                'batch_size': 32,
                'shuffle': True,
                'num_workers': 0,
                'pin_memory': False,
                'drop_last': False
            }
            
            # 合并配置
            loader_config = {**default_config, **config}
            
            # 检查是否需要特殊的collate_fn（用于文本数据）
            if hasattr(dataset, 'vocab') or 'collate_fn' in config:
                if 'collate_fn' not in config:
                    # 为文本数据使用CollateText
                    loader_config['collate_fn'] = CollateText(
                        pad_idx=getattr(dataset, 'pad_idx', 1)
                    )
            
            dataloader = DataLoader(dataset, **loader_config)
            
            self._logger.debug(f"Created dataloader with config: {loader_config}")
            return dataloader
            
        except Exception as e:
            raise FactoryError(f"Failed to create dataloader: {e}")
    
    def partition_dataset(self, 
                          dataset: Dataset, 
                          num_clients: int,
                          partition_strategy: str,
                          config: Dict[str, Any]) -> List[Dataset]:
        """
        分割数据集为多个客户端数据
        
        Args:
            dataset: 原始数据集
            num_clients: 客户端数量
            partition_strategy: 分割策略
            config: 分割配置
            
        Returns:
            分割后的数据集列表
            
        Raises:
            FactoryError: 当分割失败时
        """
        try:
            # 推断数据集类型
            dataset_type = self._infer_dataset_type(dataset)
            
            if dataset_type not in self._partition_strategies:
                raise FactoryError(f"No partition strategies available for dataset type: {dataset_type}")
            
            if partition_strategy not in self._partition_strategies[dataset_type]:
                available = self._partition_strategies[dataset_type]
                raise FactoryError(
                    f"Unsupported partition strategy '{partition_strategy}' for {dataset_type}. "
                    f"Available: {available}"
                )
            
            partition_func = self._partition_functions[dataset_type][partition_strategy]
            partitioned_datasets = partition_func(dataset, num_clients)
            
            self._logger.info(
                f"Partitioned {dataset_type} dataset into {len(partitioned_datasets)} clients "
                f"using {partition_strategy} strategy"
            )
            
            return partitioned_datasets
            
        except Exception as e:
            if isinstance(e, FactoryError):
                raise
            raise FactoryError(f"Failed to partition dataset: {e}")
    
    def create_federated_datasets(self,
                                 dataset_name: str,
                                 num_clients: int,
                                 config: Dict[str, Any]) -> Tuple[List[DataLoader], DataLoader]:
        """
        创建联邦学习数据集设置
        
        Args:
            dataset_name: 数据集名称
            num_clients: 客户端数量
            config: 配置字典
            
        Returns:
            (客户端训练数据加载器列表, 测试数据加载器)
            
        Raises:
            FactoryError: 当创建失败时
        """
        try:
            # 创建训练和测试数据集
            train_dataset = self.create_dataset(dataset_name, 'train', config)
            test_dataset = self.create_dataset(dataset_name, 'test', config)
            
            # 分割训练数据集
            partition_strategy = config.get('partition_strategy', 'label_shift')
            client_datasets = self.partition_dataset(
                train_dataset, num_clients, partition_strategy, config
            )
            
            # 创建客户端数据加载器
            dataloader_config = config.get('dataloader', {})
            client_loaders = []
            for i, client_dataset in enumerate(client_datasets):
                client_config = {**dataloader_config}
                client_config.setdefault('shuffle', True)
                client_loader = self.create_dataloader(client_dataset, client_config)
                client_loaders.append(client_loader)
                
            # 创建测试数据加载器
            test_config = {**dataloader_config}
            test_config.setdefault('shuffle', False)
            test_loader = self.create_dataloader(test_dataset, test_config)
            
            self._logger.info(
                f"Created federated setup for {dataset_name}: "
                f"{len(client_loaders)} clients, test set size: {len(test_dataset)}"
            )
            
            return client_loaders, test_loader
            
        except Exception as e:
            if isinstance(e, FactoryError):
                raise
            raise FactoryError(f"Failed to create federated datasets for {dataset_name}: {e}")
    
    def _create_mnist_dataset(self, split: str, config: Dict[str, Any]) -> Dataset:
        """创建MNIST数据集"""
        # 映射split名称
        split_mapping = {'train': 'train', 'test': 'test', 'val': 'test'}
        actual_split = split_mapping.get(split, split)
        
        # 获取缓存配置
        use_cache = config.get('use_cache', True)
        cache_dir = config.get('cache_dir', self._cache_dir)
        
        if actual_split == 'train':
            train_dataset, _ = get_mnist_datasets(
                use_cache=use_cache, 
                cache_dir=cache_dir
            )
            return train_dataset
        else:
            _, test_dataset = get_mnist_datasets(
                use_cache=use_cache,
                cache_dir=cache_dir
            )
            return test_dataset
    
    def _create_imdb_dataset(self, split: str, config: Dict[str, Any]) -> Dataset:
        """创建IMDB数据集"""
        # 映射split名称
        split_mapping = {'train': 'train', 'test': 'test', 'val': 'test'}
        actual_split = split_mapping.get(split, split)
        
        # 获取配置参数
        vocab_size = config.get('vocab_size', 10000)
        max_len = config.get('max_len', 256)
        use_cache = config.get('use_cache', True)
        cache_dir = config.get('cache_dir', self._cache_dir)
        
        datasets, vocab = get_imdb_splits(
            vocab_size=vocab_size,
            max_len=max_len,
            use_cache=use_cache,
            cache_dir=cache_dir
        )
        
        if actual_split == 'train':
            dataset = datasets['train']
        else:
            dataset = datasets['test']
            
        # 为IMDB数据集添加词汇表信息
        dataset.vocab = vocab
        dataset.vocab_size = len(vocab)
        dataset.pad_idx = vocab['<pad>']
        
        return dataset
    
    def _infer_dataset_type(self, dataset: Dataset) -> str:
        """
        推断数据集类型
        
        Args:
            dataset: 数据集实例
            
        Returns:
            数据集类型字符串
        """
        # 根据数据集属性推断类型
        if hasattr(dataset, 'vocab') or hasattr(dataset, 'pad_idx'):
            return 'imdb'
        
        # 检查数据样本结构
        if len(dataset) > 0:
            sample = dataset[0]
            if isinstance(sample, tuple) and len(sample) == 2:
                x, y = sample
                if isinstance(x, torch.Tensor):
                    if len(x.shape) == 1 and x.shape[0] == 784:  # MNIST 扁平化
                        return 'mnist'
                    elif len(x.shape) == 3 and x.shape[-2:] == (28, 28):  # MNIST 图像
                        return 'mnist'
                    elif len(x.shape) == 1:  # 文本序列
                        return 'imdb'
        
        # 默认返回未知类型
        raise FactoryError("Cannot infer dataset type")
    
    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """
        获取数据集信息
        
        Args:
            dataset_name: 数据集名称
            
        Returns:
            包含数据集信息的字典
        """
        dataset_name = dataset_name.lower()
        
        if dataset_name not in self._supported_datasets:
            raise FactoryError(f"Unknown dataset: {dataset_name}")
        
        info = {
            'name': dataset_name,
            'supported': True,
            'partition_strategies': self._partition_strategies[dataset_name]
        }
        
        if dataset_name == 'mnist':
            info.update({
                'description': 'MNIST handwritten digit classification dataset',
                'type': 'image',
                'num_classes': 10,
                'input_shape': (1, 28, 28),
                'train_samples': 60000,
                'test_samples': 10000,
                'default_config': {
                    'use_cache': True,
                    'dataloader': {
                        'batch_size': 32,
                        'shuffle': True,
                        'num_workers': 0
                    }
                }
            })
        elif dataset_name == 'imdb':
            info.update({
                'description': 'IMDB movie review sentiment classification dataset',
                'type': 'text',
                'num_classes': 2,
                'train_samples': 25000,
                'test_samples': 25000,
                'default_config': {
                    'vocab_size': 10000,
                    'max_len': 256,
                    'use_cache': True,
                    'dataloader': {
                        'batch_size': 32,
                        'shuffle': True,
                        'num_workers': 0
                    }
                }
            })
        
        return info
    
    def get_supported_partition_strategies(self, dataset_name: str) -> List[str]:
        """
        获取数据集支持的分割策略
        
        Args:
            dataset_name: 数据集名称
            
        Returns:
            支持的分割策略列表
        """
        dataset_name = dataset_name.lower()
        return self._partition_strategies.get(dataset_name, [])
    
    def validate_config(self, dataset_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        验证和规范化数据集配置
        
        Args:
            dataset_name: 数据集名称
            config: 原始配置
            
        Returns:
            验证后的配置
            
        Raises:
            FactoryError: 当配置无效时
        """
        dataset_name = dataset_name.lower()
        
        if dataset_name not in self._supported_datasets:
            raise FactoryError(f"Unsupported dataset: {dataset_name}")
        
        # 获取默认配置
        default_info = self.get_dataset_info(dataset_name)
        validated_config = default_info.get('default_config', {}).copy()
        
        # 合并用户配置
        validated_config.update(config)
        
        # 数据集特定验证
        if dataset_name == 'imdb':
            if 'vocab_size' in validated_config:
                if not isinstance(validated_config['vocab_size'], int) or validated_config['vocab_size'] <= 0:
                    raise FactoryError("vocab_size must be a positive integer")
            
            if 'max_len' in validated_config:
                if not isinstance(validated_config['max_len'], int) or validated_config['max_len'] <= 0:
                    raise FactoryError("max_len must be a positive integer")
        
        # 验证dataloader配置
        if 'dataloader' in validated_config:
            dataloader_config = validated_config['dataloader']
            if 'batch_size' in dataloader_config:
                if not isinstance(dataloader_config['batch_size'], int) or dataloader_config['batch_size'] <= 0:
                    raise FactoryError("batch_size must be a positive integer")
        
        return validated_config