"""
数据集工厂测试

测试DatasetFactory的所有功能，包括：
- 数据集创建和验证
- 数据加载器创建
- 数据集分割
- 联邦学习数据集设置创建
- 配置验证
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import torch
from torch.utils.data import Dataset, DataLoader

from src.core.exceptions import FactoryError
from src.factories.dataset_factory import DatasetFactory


class MockDataset(Dataset):
    """模拟数据集"""
    
    def __init__(self, size=100, has_vocab=False):
        self.size = size
        if has_vocab:
            self.vocab = {'<pad>': 1, '<unk>': 0}
            self.vocab_size = len(self.vocab)
            self.pad_idx = 1
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        if hasattr(self, 'vocab'):
            # IMDB风格数据
            return torch.randint(0, 100, (50,)), torch.randint(0, 2, (1,)).item()
        else:
            # MNIST风格数据
            return torch.randn(784), torch.randint(0, 10, (1,)).item()


class TestDatasetFactory(unittest.TestCase):
    """数据集工厂测试"""
    
    def setUp(self):
        """每个测试前的设置"""
        self.factory = DatasetFactory(cache_dir='test_cache')
    
    def test_initialization(self):
        """测试工厂初始化"""
        self.assertIsInstance(self.factory, DatasetFactory)
        self.assertEqual(self.factory._cache_dir, 'test_cache')
        self.assertIn('mnist', self.factory._supported_datasets)
        self.assertIn('imdb', self.factory._supported_datasets)
    
    def test_supported_datasets(self):
        """测试支持的数据集"""
        types = self.factory.list_registered_types()
        self.assertIn('mnist', types)
        self.assertIn('imdb', types)
    
    def test_is_registered(self):
        """测试检查注册状态"""
        self.assertTrue(self.factory.is_registered('mnist'))
        self.assertTrue(self.factory.is_registered('imdb'))
        self.assertFalse(self.factory.is_registered('unknown'))
    
    def test_register_and_unregister(self):
        """测试注册和注销数据集创建器"""
        # 注册成功不应该抛出异常
        self.factory.register('custom_dataset', Mock)
        
        # 注销成功不应该抛出异常
        self.factory.unregister('custom_dataset')
    
    @patch('src.datasets.get_mnist_datasets')
    def test_create_mnist_dataset(self, mock_get_mnist):
        """测试创建MNIST数据集"""
        mock_train_dataset = MockDataset(size=1000)
        mock_test_dataset = MockDataset(size=200)
        mock_get_mnist.return_value = (mock_train_dataset, mock_test_dataset)
        
        # 测试创建训练集
        train_dataset = self.factory.create_dataset('mnist', 'train', {})
        self.assertEqual(len(train_dataset), 1000)
        
        # 测试创建测试集
        test_dataset = self.factory.create_dataset('mnist', 'test', {})
        self.assertEqual(len(test_dataset), 200)
        
        # 验证调用参数
        mock_get_mnist.assert_called_with(use_cache=True, cache_dir='test_cache')
    
    @patch('src.datasets.get_imdb_splits')
    def test_create_imdb_dataset(self, mock_get_imdb):
        """测试创建IMDB数据集"""
        mock_train_dataset = MockDataset(size=1000, has_vocab=True)
        mock_test_dataset = MockDataset(size=200, has_vocab=True)
        mock_vocab = {'<pad>': 1, '<unk>': 0}
        
        mock_get_imdb.return_value = (
            {'train': mock_train_dataset, 'test': mock_test_dataset},
            mock_vocab
        )
        
        config = {'vocab_size': 5000, 'max_len': 128}
        
        # 测试创建训练集
        train_dataset = self.factory.create_dataset('imdb', 'train', config)
        self.assertEqual(len(train_dataset), 1000)
        self.assertTrue(hasattr(train_dataset, 'vocab'))
        
        # 验证调用参数
        mock_get_imdb.assert_called_with(
            vocab_size=5000,
            max_len=128,
            use_cache=True,
            cache_dir='test_cache'
        )
    
    def test_create_unsupported_dataset(self):
        """测试创建不支持的数据集"""
        with self.assertRaises(FactoryError) as cm:
            self.factory.create_dataset('unknown', 'train', {})
        
        self.assertIn('Unsupported dataset', str(cm.exception))
    
    def test_create_invalid_split(self):
        """测试创建无效分割"""
        with self.assertRaises(FactoryError) as cm:
            self.factory.create_dataset('mnist', 'invalid_split', {})
        
        self.assertIn('Invalid split', str(cm.exception))
    
    def test_create_dataloader(self):
        """测试创建数据加载器"""
        mock_dataset = MockDataset(size=100)
        
        config = {
            'batch_size': 32,
            'shuffle': True,
            'num_workers': 2
        }
        
        dataloader = self.factory.create_dataloader(mock_dataset, config)
        
        self.assertIsInstance(dataloader, DataLoader)
        self.assertEqual(dataloader.batch_size, 32)
        self.assertEqual(dataloader.num_workers, 2)
    
    def test_create_dataloader_with_text_collate_fn(self):
        """测试为文本数据创建数据加载器"""
        mock_dataset = MockDataset(size=100, has_vocab=True)
        
        config = {'batch_size': 16}
        
        dataloader = self.factory.create_dataloader(mock_dataset, config)
        
        self.assertIsInstance(dataloader, DataLoader)
        self.assertIsNotNone(dataloader.collate_fn)
    
    def test_create_dataloader_error_handling(self):
        """测试数据加载器创建错误处理"""
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(side_effect=RuntimeError("Dataset error"))
        
        config = {'batch_size': 32}
        
        with self.assertRaises(FactoryError) as cm:
            self.factory.create_dataloader(mock_dataset, config)
        
        self.assertIn('Failed to create dataloader', str(cm.exception))
    
    @patch('src.datasets.partition_mnist_label_shift')
    def test_partition_mnist_dataset(self, mock_partition_mnist):
        """测试分割MNIST数据集"""
        mock_dataset = MockDataset(size=1000)
        mock_partitioned = [MockDataset(size=100) for _ in range(10)]
        mock_partition_mnist.return_value = mock_partitioned
        
        partitioned = self.factory.partition_dataset(
            dataset=mock_dataset,
            num_clients=10,
            partition_strategy='label_shift',
            config={}
        )
        
        self.assertEqual(len(partitioned), 10)
        mock_partition_mnist.assert_called_once_with(mock_dataset, 10)
    
    @patch('src.datasets.partition_imdb_label_shift')
    def test_partition_imdb_dataset(self, mock_partition_imdb):
        """测试分割IMDB数据集"""
        mock_dataset = MockDataset(size=1000, has_vocab=True)
        mock_partitioned = [MockDataset(size=100, has_vocab=True) for _ in range(10)]
        mock_partition_imdb.return_value = mock_partitioned
        
        partitioned = self.factory.partition_dataset(
            dataset=mock_dataset,
            num_clients=10,
            partition_strategy='label_shift',
            config={}
        )
        
        self.assertEqual(len(partitioned), 10)
        mock_partition_imdb.assert_called_once_with(mock_dataset, 10)
    
    def test_partition_unsupported_strategy(self):
        """测试不支持的分割策略"""
        mock_dataset = MockDataset(size=1000)
        
        with self.assertRaises(FactoryError) as cm:
            self.factory.partition_dataset(
                dataset=mock_dataset,
                num_clients=10,
                partition_strategy='unknown_strategy',
                config={}
            )
        
        self.assertIn('Unsupported partition strategy', str(cm.exception))
    
    def test_infer_dataset_type(self):
        """测试推断数据集类型"""
        # IMDB数据集（有vocab属性）
        imdb_dataset = MockDataset(has_vocab=True)
        dataset_type = self.factory._infer_dataset_type(imdb_dataset)
        self.assertEqual(dataset_type, 'imdb')
        
        # MNIST数据集（784维向量）
        mnist_dataset = Mock()
        mnist_dataset.__len__ = Mock(return_value=1)
        mnist_dataset.__getitem__ = Mock(return_value=(torch.randn(784), 0))
        
        dataset_type = self.factory._infer_dataset_type(mnist_dataset)
        self.assertEqual(dataset_type, 'mnist')
    
    def test_infer_dataset_type_error(self):
        """测试推断数据集类型失败"""
        # 空数据集
        empty_dataset = Mock()
        empty_dataset.__len__ = Mock(return_value=0)
        
        with self.assertRaises(FactoryError):
            self.factory._infer_dataset_type(empty_dataset)
    
    @patch.multiple('src.datasets', 
                    get_mnist_datasets=Mock(), 
                    partition_mnist_label_shift=Mock())
    def test_create_federated_datasets(self):
        """测试创建联邦学习数据集设置"""
        # 设置mocks
        mock_train = MockDataset(size=1000)
        mock_test = MockDataset(size=200)
        src.datasets.get_mnist_datasets.return_value = (mock_train, mock_test)
        
        mock_partitioned = [MockDataset(size=100) for _ in range(5)]
        src.datasets.partition_mnist_label_shift.return_value = mock_partitioned
        
        config = {
            'dataloader': {'batch_size': 32},
            'partition_strategy': 'label_shift'
        }
        
        client_loaders, test_loader = self.factory.create_federated_datasets(
            dataset_name='mnist',
            num_clients=5,
            config=config
        )
        
        # 验证结果
        self.assertEqual(len(client_loaders), 5)
        for loader in client_loaders:
            self.assertIsInstance(loader, DataLoader)
            self.assertEqual(loader.batch_size, 32)
        
        self.assertIsInstance(test_loader, DataLoader)
        self.assertEqual(test_loader.batch_size, 32)
    
    def test_get_dataset_info(self):
        """测试获取数据集信息"""
        # 测试MNIST信息
        mnist_info = self.factory.get_dataset_info('mnist')
        self.assertEqual(mnist_info['name'], 'mnist')
        self.assertEqual(mnist_info['type'], 'image')
        self.assertEqual(mnist_info['num_classes'], 10)
        self.assertIn('description', mnist_info)
        self.assertIn('default_config', mnist_info)
        
        # 测试IMDB信息
        imdb_info = self.factory.get_dataset_info('imdb')
        self.assertEqual(imdb_info['name'], 'imdb')
        self.assertEqual(imdb_info['type'], 'text')
        self.assertEqual(imdb_info['num_classes'], 2)
        
        # 测试未知数据集
        with self.assertRaises(FactoryError):
            self.factory.get_dataset_info('unknown')
    
    def test_get_supported_partition_strategies(self):
        """测试获取支持的分割策略"""
        mnist_strategies = self.factory.get_supported_partition_strategies('mnist')
        self.assertIn('label_shift', mnist_strategies)
        
        imdb_strategies = self.factory.get_supported_partition_strategies('imdb')
        self.assertIn('label_shift', imdb_strategies)
        
        unknown_strategies = self.factory.get_supported_partition_strategies('unknown')
        self.assertEqual(unknown_strategies, [])
    
    def test_validate_config(self):
        """测试配置验证"""
        # 测试有效配置
        valid_config = {
            'vocab_size': 10000,
            'max_len': 256,
            'dataloader': {'batch_size': 32}
        }
        validated = self.factory.validate_config('imdb', valid_config)
        self.assertIsInstance(validated, dict)
        
        # 测试无效vocab_size
        invalid_config = {'vocab_size': -1}
        with self.assertRaises(FactoryError):
            self.factory.validate_config('imdb', invalid_config)
        
        # 测试无效batch_size
        invalid_config = {'dataloader': {'batch_size': 0}}
        with self.assertRaises(FactoryError):
            self.factory.validate_config('mnist', invalid_config)
        
        # 测试不支持的数据集
        with self.assertRaises(FactoryError):
            self.factory.validate_config('unknown', {})
    
    def test_create_with_component_type(self):
        """测试使用组件类型创建"""
        with patch('src.datasets.get_mnist_datasets') as mock_get_mnist:
            mock_train = MockDataset(size=1000)
            mock_test = MockDataset(size=200)
            mock_get_mnist.return_value = (mock_train, mock_test)
            
            config = {'split': 'train'}
            dataset = self.factory.create('mnist', config)
            
            self.assertEqual(len(dataset), 1000)
    
    def test_create_unsupported_component_type(self):
        """测试使用不支持的组件类型"""
        config = {}
        with self.assertRaises(FactoryError) as cm:
            self.factory.create('unknown', config)
        
        self.assertIn('Unsupported dataset type', str(cm.exception))
    
    def test_error_handling_in_create_federated_datasets(self):
        """测试联邦数据集创建过程中的错误处理"""
        # 模拟数据集创建失败
        with patch.object(self.factory, 'create_dataset') as mock_create:
            mock_create.side_effect = RuntimeError("Dataset creation failed")
            
            with self.assertRaises(FactoryError) as cm:
                self.factory.create_federated_datasets('mnist', 5, {})
            
            self.assertIn('Failed to create federated datasets', str(cm.exception))
    
    def test_cache_dir_inheritance(self):
        """测试缓存目录继承"""
        factory_with_cache = DatasetFactory(cache_dir='/custom/cache')
        
        # 测试缓存目录是否正确设置
        self.assertEqual(factory_with_cache._cache_dir, '/custom/cache')
        
        # 测试创建数据集时是否使用了缓存目录
        with patch('src.datasets.get_mnist_datasets') as mock_get_mnist:
            mock_train = MockDataset(size=1000)
            mock_test = MockDataset(size=200)
            mock_get_mnist.return_value = (mock_train, mock_test)
            
            config = {}
            dataset = factory_with_cache.create_dataset('mnist', 'train', config)
            
            # 验证get_mnist_datasets被调用时使用了正确的cache_dir
            mock_get_mnist.assert_called_with(use_cache=True, cache_dir='/custom/cache')


if __name__ == '__main__':
    unittest.main()