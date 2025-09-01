"""
模型工厂测试

测试ModelFactory的所有功能，包括：
- 模型创建和验证
- 配置处理和验证
- 错误处理
- 模型信息获取
- 默认配置使用
"""

import unittest
from unittest.mock import patch, Mock
import torch.nn as nn

from src.core.exceptions import FactoryError
from src.factories.model_factory import ModelFactory


class TestModelFactory(unittest.TestCase):
    """模型工厂测试"""
    
    def setUp(self):
        """每个测试前的设置"""
        self.factory = ModelFactory()
    
    def test_initialization(self):
        """测试工厂初始化"""
        self.assertIsInstance(self.factory, ModelFactory)
        self.assertIn('mnist', self.factory.get_supported_datasets())
        self.assertIn('imdb', self.factory.get_supported_datasets())
    
    def test_supported_datasets(self):
        """测试支持的数据集"""
        datasets = self.factory.get_supported_datasets()
        self.assertIn('mnist', datasets)
        self.assertIn('imdb', datasets)
        self.assertIsInstance(datasets, list)
    
    def test_supported_models(self):
        """测试支持的模型"""
        mnist_models = self.factory.get_supported_models('mnist')
        self.assertIn('mlp', mnist_models)
        self.assertIn('vit', mnist_models)
        
        imdb_models = self.factory.get_supported_models('imdb')
        self.assertIn('rnn', imdb_models)
        self.assertIn('lstm', imdb_models)
        self.assertIn('transformer', imdb_models)
        
        # 测试不存在的数据集
        unknown_models = self.factory.get_supported_models('unknown')
        self.assertEqual(unknown_models, [])
    
    def test_create_mnist_mlp(self):
        """测试创建MNIST MLP模型"""
        config = {
            'input_size': 784,
            'hidden_sizes': [512, 256],
            'num_classes': 10
        }
        
        model = self.factory.create_model('mnist', 'mlp', config)
        self.assertIsInstance(model, nn.Module)
        
        # 测试模型结构
        self.assertTrue(hasattr(model, 'mlp'))
    
    def test_create_mnist_vit(self):
        """测试创建MNIST Vision Transformer模型"""
        config = {
            'image_size': 28,
            'patch_size': 7,
            'emb_dim': 128,
            'depth': 4,
            'nhead': 4,
            'num_classes': 10
        }
        
        model = self.factory.create_model('mnist', 'vit', config)
        self.assertIsInstance(model, nn.Module)
    
    def test_create_imdb_rnn(self):
        """测试创建IMDB RNN模型"""
        config = {
            'embedding_dim': 128,
            'hidden_size': 128,
            'num_layers': 1,
            'bidirectional': False
        }
        extra = {'vocab_size': 10000}
        
        model = self.factory.create_model('imdb', 'rnn', config, **extra)
        self.assertIsInstance(model, nn.Module)
        
        # 测试模型结构
        self.assertTrue(hasattr(model, 'embedding'))
        self.assertTrue(hasattr(model, 'rnn'))
        self.assertTrue(hasattr(model, 'classifier'))
    
    def test_create_imdb_lstm(self):
        """测试创建IMDB LSTM模型"""
        config = {
            'embedding_dim': 128,
            'hidden_size': 128,
            'num_layers': 1,
            'bidirectional': False
        }
        extra = {'vocab_size': 10000}
        
        model = self.factory.create_model('imdb', 'lstm', config, **extra)
        self.assertIsInstance(model, nn.Module)
    
    def test_create_imdb_transformer(self):
        """测试创建IMDB Transformer模型"""
        config = {
            'embedding_dim': 128,
            'nhead': 4,
            'num_layers': 2,
            'hidden_dim': 256,
            'max_seq_len': 256
        }
        extra = {'vocab_size': 10000}
        
        model = self.factory.create_model('imdb', 'transformer', config, **extra)
        self.assertIsInstance(model, nn.Module)
    
    def test_create_unsupported_dataset(self):
        """测试创建不支持的数据集模型"""
        config = {}
        
        with self.assertRaises(FactoryError) as cm:
            self.factory.create_model('unknown', 'mlp', config)
        
        self.assertIn('Unsupported dataset', str(cm.exception))
    
    def test_create_unsupported_model(self):
        """测试创建不支持的模型类型"""
        config = {}
        
        with self.assertRaises(FactoryError) as cm:
            self.factory.create_model('mnist', 'unknown', config)
        
        self.assertIn('Unsupported model', str(cm.exception))
    
    def test_create_imdb_without_vocab_size(self):
        """测试创建IMDB模型但缺少vocab_size"""
        config = {'embedding_dim': 128}
        
        with self.assertRaises(FactoryError) as cm:
            self.factory.create_model('imdb', 'rnn', config)
        
        self.assertIn('vocab_size', str(cm.exception))
    
    def test_create_with_component_type(self):
        """测试使用组件类型创建模型"""
        config = {
            'input_size': 784,
            'hidden_sizes': [256],
            'num_classes': 10
        }
        
        model = self.factory.create('mnist_mlp', config)
        self.assertIsInstance(model, nn.Module)
    
    def test_create_with_invalid_component_type(self):
        """测试使用无效组件类型"""
        config = {}
        
        with self.assertRaises(FactoryError) as cm:
            self.factory.create('invalid', config)
        
        self.assertIn('Invalid model component type', str(cm.exception))
    
    def test_register_and_unregister(self):
        """测试注册和注销模型创建器"""
        # 注册成功不应该抛出异常
        self.factory.register('custom_model', Mock)
        
        # 注销成功不应该抛出异常
        self.factory.unregister('custom_model')
    
    def test_list_registered_types(self):
        """测试列出已注册的类型"""
        types = self.factory.list_registered_types()
        
        # 应该包含内置类型
        self.assertIn('mnist_mlp', types)
        self.assertIn('mnist_vit', types)
        self.assertIn('imdb_rnn', types)
        self.assertIn('imdb_lstm', types)
        self.assertIn('imdb_transformer', types)
    
    def test_is_registered(self):
        """测试检查注册状态"""
        # 内置模型应该被注册
        self.assertTrue(self.factory.is_registered('mnist_mlp'))
        self.assertTrue(self.factory.is_registered('imdb_rnn'))
        
        # 不存在的模型应该未注册
        self.assertFalse(self.factory.is_registered('unknown_model'))
    
    def test_config_validation_mnist_mlp(self):
        """测试MNIST MLP配置验证"""
        # 无效的hidden_sizes类型
        config = {'hidden_sizes': 'invalid'}
        
        with self.assertRaises(FactoryError):
            self.factory.create_model('mnist', 'mlp', config)
    
    def test_config_validation_mnist_vit(self):
        """测试MNIST Vision Transformer配置验证"""
        # patch_size不能整除image_size
        config = {
            'image_size': 28,
            'patch_size': 5,  # 28不能被5整除
            'emb_dim': 128,
            'nhead': 4
        }
        
        with self.assertRaises(FactoryError):
            self.factory.create_model('mnist', 'vit', config)
        
        # emb_dim不能被nhead整除
        config = {
            'image_size': 28,
            'patch_size': 7,
            'emb_dim': 127,  # 127不能被4整除
            'nhead': 4
        }
        
        with self.assertRaises(FactoryError):
            self.factory.create_model('mnist', 'vit', config)
    
    def test_config_validation_imdb_models(self):
        """测试IMDB模型配置验证"""
        config = {'embedding_dim': 128}
        extra = {'vocab_size': 0}  # 无效的vocab_size
        
        with self.assertRaises(FactoryError):
            self.factory.create_model('imdb', 'rnn', config, **extra)
        
        # 测试Transformer特有验证
        config = {
            'embedding_dim': 127,  # 不能被nhead整除
            'nhead': 4
        }
        extra = {'vocab_size': 10000}
        
        with self.assertRaises(FactoryError):
            self.factory.create_model('imdb', 'transformer', config, **extra)
    
    def test_get_model_info(self):
        """测试获取模型信息"""
        # 测试MNIST MLP信息
        info = self.factory.get_model_info('mnist', 'mlp')
        self.assertEqual(info['dataset'], 'mnist')
        self.assertEqual(info['model_type'], 'mlp')
        self.assertIn('description', info)
        self.assertIn('default_config', info)
        
        # 测试IMDB RNN信息
        info = self.factory.get_model_info('imdb', 'rnn')
        self.assertEqual(info['dataset'], 'imdb')
        self.assertEqual(info['model_type'], 'rnn')
        self.assertTrue(info['requires_vocab_size'])
        
        # 测试未知模型
        with self.assertRaises(FactoryError):
            self.factory.get_model_info('unknown', 'model')
    
    def test_create_model_with_defaults(self):
        """测试使用默认配置创建模型"""
        # 测试MNIST MLP默认配置
        model = self.factory.create_model_with_defaults('mnist', 'mlp')
        self.assertIsInstance(model, nn.Module)
        
        # 测试IMDB RNN默认配置
        model = self.factory.create_model_with_defaults(
            'imdb', 'rnn', vocab_size=10000
        )
        self.assertIsInstance(model, nn.Module)
        
        # 测试配置覆盖
        model = self.factory.create_model_with_defaults(
            'mnist', 'mlp', 
            config_overrides={'hidden_sizes': [128]},
        )
        self.assertIsInstance(model, nn.Module)
    
    def test_case_insensitive_creation(self):
        """测试大小写不敏感的创建"""
        config = {
            'input_size': 784,
            'hidden_sizes': [256],
            'num_classes': 10
        }
        
        # 测试不同大小写组合
        model1 = self.factory.create_model('MNIST', 'MLP', config)
        model2 = self.factory.create_model('mnist', 'mlp', config)
        model3 = self.factory.create_model('Mnist', 'Mlp', config)
        
        # 都应该成功创建
        self.assertIsInstance(model1, nn.Module)
        self.assertIsInstance(model2, nn.Module)
        self.assertIsInstance(model3, nn.Module)
    
    @patch('src.factories.model_factory.legacy_create_model')
    def test_legacy_integration(self, mock_create_model):
        """测试与现有模型创建逻辑的集成"""
        mock_model = Mock(spec=nn.Module)
        mock_create_model.return_value = mock_model
        
        config = {'input_size': 784}
        model = self.factory.create_model('mnist', 'mlp', config)
        
        # 验证调用了原始的create_model函数
        mock_create_model.assert_called_once_with('mnist', 'mlp', config, {})
        self.assertEqual(model, mock_model)
    
    def test_error_handling_in_create(self):
        """测试创建过程中的错误处理"""
        # 模拟原始create_model函数抛出异常
        with patch('src.factories.model_factory.legacy_create_model') as mock_create_model:
            mock_create_model.side_effect = RuntimeError("Model creation failed")
            
            config = {}
            with self.assertRaises(FactoryError) as cm:
                self.factory.create_model('mnist', 'mlp', config)
            
            self.assertIn('Failed to create model', str(cm.exception))
            self.assertIn('Model creation failed', str(cm.exception))


if __name__ == '__main__':
    unittest.main()