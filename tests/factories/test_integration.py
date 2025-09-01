"""
工厂系统集成测试

验证工厂模式与现有策略系统的集成，以及完整的组件创建流程。
"""

import unittest
from unittest.mock import patch, Mock

from src.factories import ComponentFactory
from src.core.exceptions import FactoryError
from src.strategies import StrategyManager


class TestFactoryIntegration(unittest.TestCase):
    """工厂系统集成测试"""
    
    def setUp(self):
        """测试前设置"""
        self.factory = ComponentFactory()
    
    def test_factory_initialization(self):
        """测试工厂初始化"""
        self.assertIsNotNone(self.factory.get_model_factory())
        self.assertIsNotNone(self.factory.get_dataset_factory())
        self.assertIsNotNone(self.factory.get_client_factory())
        self.assertIsNotNone(self.factory.get_server_factory())
        self.assertIsNotNone(self.factory.get_aggregator_factory())
    
    def test_strategy_system_integration(self):
        """测试与策略系统的集成"""
        aggregator_factory = self.factory.get_aggregator_factory()
        
        # 应该返回StrategyManager实例
        self.assertIsInstance(aggregator_factory, StrategyManager)
    
    def test_get_supported_configurations(self):
        """测试获取支持的配置"""
        config_info = self.factory.get_supported_configurations()
        
        self.assertIn('datasets', config_info)
        self.assertIn('mnist', config_info['datasets'])
        self.assertIn('imdb', config_info['datasets'])
        
        # MNIST应该支持MLP和ViT
        mnist_models = config_info['datasets']['mnist']['supported_models']
        self.assertIn('mlp', mnist_models)
        self.assertIn('vit', mnist_models)
        
        # IMDB应该支持RNN、LSTM、Transformer
        imdb_models = config_info['datasets']['imdb']['supported_models']
        self.assertIn('rnn', imdb_models)
        self.assertIn('lstm', imdb_models)
        self.assertIn('transformer', imdb_models)
    
    def test_validate_full_config(self):
        """测试完整配置验证"""
        # 测试有效配置
        valid_config = {
            'dataset': 'mnist',
            'model_type': 'mlp',
            'num_clients': 10,
            'device': 'cpu'
        }
        
        report = self.factory.validate_full_config(valid_config)
        self.assertTrue(report['valid'])
        self.assertEqual(len(report['errors']), 0)
        
        # 测试无效配置
        invalid_config = {
            'dataset': 'unknown_dataset',
            'model_type': 'mlp',
            'num_clients': 10
        }
        
        report = self.factory.validate_full_config(invalid_config)
        self.assertFalse(report['valid'])
        self.assertGreater(len(report['errors']), 0)
    
    def test_config_compatibility_checks(self):
        """测试配置兼容性检查"""
        # 测试LoRA和AdaLoRA冲突
        config_with_conflict = {
            'dataset': 'mnist',
            'model_type': 'mlp',
            'num_clients': 10,
            'lora': {'replaced_modules': ['fc']},
            'adalora': {'replaced_modules': ['fc']}
        }
        
        report = self.factory.validate_full_config(config_with_conflict)
        self.assertFalse(report['valid'])
        self.assertTrue(
            any('LoRA' in error and 'AdaLoRA' in error for error in report['errors'])
        )
    
    def test_factory_stats(self):
        """测试工厂统计信息"""
        stats = self.factory.get_factory_stats()
        
        self.assertIn('component_factory', stats)
        self.assertIn('model_factory', stats)
        self.assertIn('dataset_factory', stats)
        self.assertIn('client_factory', stats)
        self.assertIn('server_factory', stats)
        self.assertIn('registry', stats)
        
        # 检查subfactories列表
        subfactories = stats['component_factory']['subfactories']
        expected_subfactories = ['model', 'dataset', 'client', 'server']
        for factory in expected_subfactories:
            self.assertIn(factory, subfactories)
    
    def test_quick_setup(self):
        """测试快速设置功能的配置验证部分"""
        # 测试配置创建和验证（不需要实际的数据集操作）
        try:
            config = {
                'dataset': 'mnist',
                'model_type': 'mlp',
                'num_clients': 5,
                'device': 'cpu'
            }
            
            # 验证配置规范化
            validated_config = self.factory._validate_and_normalize_config(config)
            
            # 检查配置字段
            self.assertEqual(validated_config['dataset'], 'mnist')
            self.assertEqual(validated_config['model_type'], 'mlp')
            self.assertEqual(validated_config['num_clients'], 5)
            self.assertEqual(validated_config['device'], 'cpu')
            
            # 检查默认配置
            self.assertIn('dataset_config', validated_config)
            self.assertIn('model_config', validated_config)
            self.assertIn('client_config', validated_config)
            self.assertIn('server_config', validated_config)
            
        except Exception as e:
            self.fail(f"Quick setup config validation failed: {e}")
    
    @patch('src.datasets.get_mnist_datasets')
    def test_model_constructor_creation(self, mock_get_mnist):
        """测试模型构造器创建"""
        # 设置mock
        mock_get_mnist.return_value = (Mock(), Mock())
        
        # 创建模型构造器
        config = {
            'dataset': 'mnist',
            'model_type': 'mlp',
            'model_config': {'hidden_sizes': [256, 128]},
            'num_clients': 2
        }
        
        validated_config = self.factory._validate_and_normalize_config(config)
        model_constructor = self.factory._create_model_constructor(
            'mnist', 'mlp', validated_config
        )
        
        # 测试构造器功能
        model = model_constructor()
        self.assertIsNotNone(model)
    
    def test_run_name_generation(self):
        """测试运行名称生成"""
        # 测试自定义名称
        config_with_name = {'run_name': 'custom_experiment'}
        name = self.factory._generate_run_name(config_with_name)
        self.assertEqual(name, 'custom_experiment')
        
        # 测试自动生成（标准模式）
        config_standard = {
            'dataset': 'mnist',
            'model_type': 'mlp'
        }
        name = self.factory._generate_run_name(config_standard)
        self.assertTrue(name.startswith('mnist_mlp_standard_'))
        
        # 测试LoRA模式
        config_lora = {
            'dataset': 'imdb',
            'model_type': 'transformer',
            'lora': {'replaced_modules': ['attention']}
        }
        name = self.factory._generate_run_name(config_lora)
        self.assertTrue(name.startswith('imdb_transformer_lora_'))
        
        # 测试AdaLoRA模式
        config_adalora = {
            'dataset': 'mnist',
            'model_type': 'vit',
            'adalora': {'replaced_modules': ['attention']}
        }
        name = self.factory._generate_run_name(config_adalora)
        self.assertTrue(name.startswith('mnist_vit_adalora_'))
    
    def test_factory_registry_integration(self):
        """测试工厂注册系统集成"""
        registry = self.factory.get_factory_registry()
        
        # 获取已注册的工厂（如果为空则初始化工厂会自动注册）
        registered_factories = registry.list_registered_factories()
        
        # 如果列表为空，说明注册器被清空了，重新创建factory来触发注册
        if not registered_factories:
            # 重新导入模块来触发装饰器注册
            import importlib
            import src.factories.model_factory
            import src.factories.dataset_factory  
            import src.factories.client_factory
            import src.factories.server_factory
            import src.factories.component_factory
            
            importlib.reload(src.factories.model_factory)
            importlib.reload(src.factories.dataset_factory)
            importlib.reload(src.factories.client_factory)
            importlib.reload(src.factories.server_factory)
            importlib.reload(src.factories.component_factory)
            
            registered_factories = registry.list_registered_factories()
        
        # 现在应该有注册的工厂
        self.assertGreater(len(registered_factories), 0, "No factories registered")
        
        # 验证核心工厂类型存在（至少应该有一些）
        self.assertTrue(any(factory in registered_factories for factory in 
                           ['model', 'dataset', 'client', 'server', 'component']),
                       f"Expected core factories not found in {registered_factories}")
    
    def test_error_propagation(self):
        """测试错误传播"""
        # 测试配置验证错误
        invalid_config = {}  # 缺少必需字段
        
        with self.assertRaises(FactoryError) as cm:
            self.factory.create_federated_setup(invalid_config)
        
        error_message = str(cm.exception)
        self.assertIn('Missing required field', error_message)


if __name__ == '__main__':
    unittest.main()