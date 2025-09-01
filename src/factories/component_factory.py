"""
主工厂实现

统一管理所有子工厂，提供一站式的组件创建服务。
支持完整的联邦学习设置创建和配置管理。
"""

from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
from torch.utils.data import DataLoader

from ..core.interfaces.factory import ComponentFactoryInterface
from ..core.exceptions import FactoryError
from .factory_registry import get_factory_registry, register_factory
from .model_factory import ModelFactory
from .dataset_factory import DatasetFactory
from .client_factory import ClientFactory
from .server_factory import ServerFactory


@register_factory("component")
class ComponentFactory(ComponentFactoryInterface):
    """
    主工厂实现
    
    统一管理所有子工厂，提供一站式的联邦学习组件创建服务。
    支持从配置文件直接创建完整的联邦学习环境。
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        self._logger = logging.getLogger(__name__)
        self._registry = get_factory_registry()
        self._cache_dir = cache_dir
        
        # 初始化子工厂
        self._model_factory = ModelFactory()
        self._dataset_factory = DatasetFactory(cache_dir=cache_dir)
        self._client_factory = ClientFactory()
        self._server_factory = ServerFactory()
        
        self._logger.info(f"ComponentFactory initialized with cache_dir: {cache_dir}")
    
    def get_model_factory(self) -> ModelFactory:
        """获取模型工厂"""
        return self._model_factory
    
    def get_dataset_factory(self) -> DatasetFactory:
        """获取数据集工厂"""
        return self._dataset_factory
    
    def get_client_factory(self) -> ClientFactory:
        """获取客户端工厂"""
        return self._client_factory
    
    def get_server_factory(self) -> ServerFactory:
        """获取服务器工厂"""
        return self._server_factory
    
    def get_aggregator_factory(self):
        """获取聚合器工厂（通过策略系统）"""
        # 聚合器通过策略系统管理，这里返回策略管理器
        from ..strategies import StrategyManager
        return StrategyManager()
    
    def create_federated_setup(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        创建完整的联邦学习设置
        
        Args:
            config: 完整的配置字典
            
        Returns:
            包含所有组件的字典：
            {
                'model_constructor': 模型构造函数,
                'train_loaders': 客户端训练数据加载器列表,
                'test_loader': 测试数据加载器,
                'clients': 客户端列表,
                'server': 服务器实例,
                'config': 处理后的配置
            }
            
        Raises:
            FactoryError: 当创建失败时
        """
        try:
            self._logger.info("Creating complete federated learning setup...")
            
            # 验证和规范化配置
            validated_config = self._validate_and_normalize_config(config)
            
            # 提取关键配置
            dataset_name = validated_config['dataset']
            model_type = validated_config['model_type']
            num_clients = validated_config['num_clients']
            
            # 1. 创建数据集和数据加载器
            self._logger.info(f"Creating datasets: {dataset_name}")
            client_loaders, test_loader = self._dataset_factory.create_federated_datasets(
                dataset_name=dataset_name,
                num_clients=num_clients,
                config=validated_config.get('dataset_config', {})
            )
            
            # 2. 创建模型构造函数
            self._logger.info(f"Creating model constructor: {model_type}")
            model_constructor = self._create_model_constructor(
                dataset_name, model_type, validated_config
            )
            
            # 3. 创建客户端
            self._logger.info(f"Creating {num_clients} clients")
            clients = self._client_factory.create_clients_batch(
                dataloaders=client_loaders,
                config={
                    **validated_config.get('client_config', {}),
                    'model_constructor': model_constructor,
                    'device': validated_config.get('device', 'cpu')
                }
            )
            
            # 4. 创建服务器
            self._logger.info("Creating server")
            server = self._server_factory.create_server(
                model_constructor=model_constructor,
                clients=clients,
                config={
                    **validated_config.get('server_config', {}),
                    'run_name': self._generate_run_name(validated_config),
                    'dataset': dataset_name,
                    'model_type': model_type,
                    'device': validated_config.get('device', 'cpu'),
                    'lora_cfg': validated_config.get('lora', {}),
                    'adalora_cfg': validated_config.get('adalora', {}),
                    'model_info': {
                        'dataset': dataset_name,
                        'model_type': model_type
                    }
                }
            )
            
            # 5. 组装结果
            result = {
                'model_constructor': model_constructor,
                'train_loaders': client_loaders,
                'test_loader': test_loader,
                'clients': clients,
                'server': server,
                'config': validated_config
            }
            
            self._logger.info("Federated learning setup created successfully")
            return result
            
        except Exception as e:
            if isinstance(e, FactoryError):
                raise
            raise FactoryError(f"Failed to create federated setup: {e}")
    
    def _validate_and_normalize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        验证和规范化配置
        
        Args:
            config: 原始配置
            
        Returns:
            验证后的配置
            
        Raises:
            FactoryError: 当配置无效时
        """
        # 必需的配置项
        required_fields = ['dataset', 'model_type', 'num_clients']
        for field in required_fields:
            if field not in config:
                raise FactoryError(f"Missing required field: {field}")
        
        # 验证数据集和模型
        dataset = config['dataset'].lower()
        model_type = config['model_type'].lower()
        
        if not self._model_factory.get_supported_datasets().__contains__(dataset):
            available = self._model_factory.get_supported_datasets()
            raise FactoryError(f"Unsupported dataset: {dataset}. Available: {available}")
        
        if model_type not in self._model_factory.get_supported_models(dataset):
            available = self._model_factory.get_supported_models(dataset)
            raise FactoryError(f"Unsupported model '{model_type}' for dataset '{dataset}'. Available: {available}")
        
        # 验证客户端数量
        num_clients = config['num_clients']
        if not isinstance(num_clients, int) or num_clients <= 0:
            raise FactoryError("num_clients must be a positive integer")
        
        # 创建规范化配置
        normalized_config = {
            'dataset': dataset,
            'model_type': model_type,
            'num_clients': num_clients,
            'device': config.get('device', 'cpu'),
            'run_name': config.get('run_name'),
        }
        
        # 处理子配置
        normalized_config['dataset_config'] = self._normalize_dataset_config(dataset, config)
        normalized_config['model_config'] = config.get('model', {})
        normalized_config['client_config'] = config.get('client', {})
        normalized_config['server_config'] = config.get('server', {})
        normalized_config['lora'] = config.get('lora', {})
        normalized_config['adalora'] = config.get('adalora', {})
        
        return normalized_config
    
    def _normalize_dataset_config(self, dataset: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """规范化数据集配置"""
        dataset_config = config.get('dataset_config', {})
        
        # 为IMDB数据集添加特殊处理
        if dataset == 'imdb':
            if 'vocab_size' not in dataset_config:
                # 从模型配置或全局配置中获取vocab_size
                vocab_size = config.get('vocab_size')
                if vocab_size is None:
                    model_config = config.get('model', {})
                    vocab_size = model_config.get('vocab_size', 10000)
                dataset_config['vocab_size'] = vocab_size
        
        # 合并其他配置
        if 'dataloader' not in dataset_config:
            dataset_config['dataloader'] = {}
        
        # 从全局配置中继承
        if 'batch_size' in config:
            dataset_config['dataloader']['batch_size'] = config['batch_size']
        
        if self._cache_dir:
            dataset_config['cache_dir'] = self._cache_dir
        
        return dataset_config
    
    def _create_model_constructor(self, 
                                 dataset: str, 
                                 model_type: str, 
                                 config: Dict[str, Any]) -> callable:
        """
        创建模型构造函数
        
        Args:
            dataset: 数据集名称
            model_type: 模型类型
            config: 配置字典
            
        Returns:
            模型构造函数
        """
        model_config = config.get('model_config', {})
        
        def model_constructor(**kwargs):
            # 合并配置和运行时参数
            full_config = {**model_config, **kwargs}
            extra = {}
            
            # 为IMDB模型添加vocab_size
            if dataset == 'imdb':
                vocab_size = config['dataset_config'].get('vocab_size', 10000)
                extra['vocab_size'] = vocab_size
            
            return self._model_factory.create_model(
                dataset, model_type, full_config, **extra
            )
        
        return model_constructor
    
    def _generate_run_name(self, config: Dict[str, Any]) -> str:
        """生成运行名称"""
        if config.get('run_name'):
            return config['run_name']
        
        # 自动生成运行名称
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset = config['dataset']
        model_type = config['model_type']
        
        # 检测训练模式
        if config.get('adalora') and config['adalora'].get('replaced_modules'):
            mode = 'adalora'
        elif config.get('lora') and config['lora'].get('replaced_modules'):
            mode = 'lora'
        else:
            mode = 'standard'
        
        return f"{dataset}_{model_type}_{mode}_{timestamp}"
    
    def create_quick_setup(self, 
                          dataset: str, 
                          model_type: str, 
                          num_clients: int = 10,
                          device: str = 'cpu',
                          **kwargs) -> Dict[str, Any]:
        """
        快速创建联邦学习设置
        
        Args:
            dataset: 数据集名称
            model_type: 模型类型
            num_clients: 客户端数量
            device: 计算设备
            **kwargs: 其他配置参数
            
        Returns:
            包含所有组件的字典
        """
        config = {
            'dataset': dataset,
            'model_type': model_type,
            'num_clients': num_clients,
            'device': device,
            **kwargs
        }
        
        return self.create_federated_setup(config)
    
    def get_factory_registry(self):
        """获取工厂注册器"""
        return self._registry
    
    def get_supported_configurations(self) -> Dict[str, Any]:
        """
        获取支持的配置信息
        
        Returns:
            支持的配置信息
        """
        datasets = self._model_factory.get_supported_datasets()
        config_info = {
            'datasets': {}
        }
        
        for dataset in datasets:
            models = self._model_factory.get_supported_models(dataset)
            config_info['datasets'][dataset] = {
                'supported_models': models,
                'info': self._dataset_factory.get_dataset_info(dataset),
                'model_details': {}
            }
            
            for model in models:
                config_info['datasets'][dataset]['model_details'][model] = \
                    self._model_factory.get_model_info(dataset, model)
        
        return config_info
    
    def validate_full_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        验证完整配置
        
        Args:
            config: 配置字典
            
        Returns:
            验证报告
        """
        report = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'suggestions': []
        }
        
        try:
            validated_config = self._validate_and_normalize_config(config)
            report['normalized_config'] = validated_config
        except FactoryError as e:
            report['valid'] = False
            report['errors'].append(str(e))
        
        # 检查配置兼容性
        if report['valid']:
            self._check_config_compatibility(validated_config, report)
        
        return report
    
    def _check_config_compatibility(self, config: Dict[str, Any], report: Dict[str, Any]) -> None:
        """检查配置兼容性"""
        dataset = config['dataset']
        model_type = config['model_type']
        
        # 检查IMDB模型的vocab_size
        if dataset == 'imdb':
            vocab_size = config['dataset_config'].get('vocab_size')
            if vocab_size and vocab_size > 50000:
                report['warnings'].append(
                    f"Large vocab_size ({vocab_size}) may cause memory issues"
                )
        
        # 检查LoRA和AdaLoRA配置冲突
        lora_enabled = config.get('lora', {}).get('replaced_modules')
        adalora_enabled = config.get('adalora', {}).get('replaced_modules')
        
        if lora_enabled and adalora_enabled:
            report['errors'].append("Cannot enable both LoRA and AdaLoRA simultaneously")
            report['valid'] = False
        
        # 检查设备配置
        device = config.get('device', 'cpu')
        if device.startswith('cuda') and not hasattr(__import__('torch'), 'cuda'):
            report['warnings'].append("CUDA device specified but torch.cuda not available")
        
        # 建议优化
        num_clients = config['num_clients']
        if dataset in ['mnist', 'imdb'] and num_clients != 10:
            report['suggestions'].append(
                f"For {dataset}, using {num_clients} clients. "
                f"Consider using 10 clients for optimal non-IID distribution"
            )
    
    def get_factory_stats(self) -> Dict[str, Any]:
        """
        获取工厂统计信息
        
        Returns:
            包含统计信息的字典
        """
        return {
            'component_factory': {
                'cache_dir': self._cache_dir,
                'subfactories': ['model', 'dataset', 'client', 'server']
            },
            'model_factory': self._model_factory.get_supported_datasets(),
            'dataset_factory': self._dataset_factory.list_registered_types(),
            'client_factory': self._client_factory.get_factory_stats(),
            'server_factory': self._server_factory.get_factory_stats(),
            'registry': self._registry.get_registry_stats()
        }