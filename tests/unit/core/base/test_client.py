"""
AbstractClient 抽象类全面单元测试
"""

import unittest
from unittest.mock import Mock, MagicMock, patch
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.core.base.client import AbstractClient
from src.core.base.component import ComponentStatus
from src.core.exceptions import (
    ClientConfigurationError, 
    ClientTrainingError,
    ConfigValidationError
)


class ConcreteClient(AbstractClient):
    """用于测试的具体客户端实现"""
    
    def __init__(self, *args, **kwargs):
        # 为测试创建简单的数据加载器
        if 'train_data_loader' not in kwargs:
            dataset = TensorDataset(torch.randn(100, 10), torch.randint(0, 2, (100,)))
            kwargs['train_data_loader'] = DataLoader(dataset, batch_size=10)
        super().__init__(*args, **kwargs)
        
    def _validate_config(self):
        # 基本配置验证
        if not self._config.get('optimizer'):
            raise ClientConfigurationError("optimizer config required")
        if not isinstance(self._config['optimizer'], dict):
            raise ClientConfigurationError("optimizer must be a dict")
            
    def receive_global_model(self, global_model_state):
        # 简单的实现用于测试
        self._received_model_state = global_model_state
        
    def local_train(self, num_epochs):
        # Mock实现返回测试数据
        if not hasattr(self, '_received_model_state'):
            raise ClientTrainingError("No global model received")
            
        model_state = {'param1': torch.randn(10, 10)}
        metrics = {
            'train_acc': 0.85,
            'train_f1': 0.82,
            'train_loss': 0.35
        }
        return model_state, metrics, self.num_train_samples
        
    def local_evaluate(self, model_state=None):
        # Mock实现返回测试指标
        return {
            'test_acc': 0.78,
            'test_f1': 0.75,
            'test_loss': 0.42
        }


class TestAbstractClient(unittest.TestCase):
    
    def setUp(self):
        """测试前设置"""
        self.config = {
            'optimizer': {
                'name': 'adam',
                'lr': 0.001,
                'weight_decay': 0.0001
            },
            'device': 'cpu'
        }
        
        # 创建测试数据集
        self.dataset = TensorDataset(
            torch.randn(50, 10), 
            torch.randint(0, 2, (50,))
        )
        self.train_loader = DataLoader(self.dataset, batch_size=5)
        
    def test_client_initialization_success(self):
        """测试客户端成功初始化"""
        client = ConcreteClient(
            client_id=1,
            train_data_loader=self.train_loader,
            config=self.config,
            device='cpu'
        )
        
        # 验证基本属性
        self.assertEqual(client.client_id, 1)
        self.assertEqual(client.device, 'cpu')
        self.assertEqual(client.component_id, 'client_1')
        self.assertEqual(client.status, ComponentStatus.READY)
        self.assertEqual(client.num_train_samples, 50)
        
    def test_client_initialization_with_invalid_config(self):
        """测试无效配置时初始化失败"""
        invalid_config = {'device': 'cpu'}  # 缺少optimizer
        
        with self.assertRaises(ClientConfigurationError):
            ConcreteClient(
                client_id=1,
                train_data_loader=self.train_loader,
                config=invalid_config
            )
            
    def test_data_distribution_info(self):
        """测试数据分布信息获取"""
        client = ConcreteClient(
            client_id=1,
            train_data_loader=self.train_loader,
            config=self.config
        )
        
        info = client.get_data_distribution_info()
        
        self.assertEqual(info['train_samples'], 50)
        self.assertEqual(info['test_samples'], 0)  # 默认无测试集
        self.assertEqual(info['total_samples'], 50)
        
    def test_test_dataloader_management(self):
        """测试测试数据加载器管理"""
        client = ConcreteClient(
            client_id=1,
            train_data_loader=self.train_loader,
            config=self.config
        )
        
        # 初始时没有测试数据加载器
        self.assertIsNone(client.test_data_loader)
        self.assertEqual(client.num_test_samples, 0)
        
        # 设置测试数据加载器
        test_dataset = TensorDataset(torch.randn(20, 10), torch.randint(0, 2, (20,)))
        test_loader = DataLoader(test_dataset, batch_size=5)
        
        client.set_test_data_loader(test_loader)
        
        self.assertIsNotNone(client.test_data_loader)
        self.assertEqual(client.num_test_samples, 20)
        
        # 更新数据分布信息
        info = client.get_data_distribution_info()
        self.assertEqual(info['test_samples'], 20)
        self.assertEqual(info['total_samples'], 70)
        
    def test_local_train_and_evaluate_flow(self):
        """测试完整的训练评估流程"""
        client = ConcreteClient(
            client_id=1,
            train_data_loader=self.train_loader,
            config=self.config
        )
        
        # 模拟全局模型状态
        global_model_state = {'global_param': torch.randn(5, 5)}
        
        # 执行训练和评估
        model_state, metrics, num_samples = client.local_train_and_evaluate(
            global_model_state, num_epochs=2
        )
        
        # 验证返回结果
        self.assertIsInstance(model_state, dict)
        self.assertIn('param1', model_state)
        
        # 验证指标
        expected_keys = [
            'train_acc', 'train_f1', 'train_loss',
            'test_acc', 'test_f1', 'test_loss',
            'round', 'client_id', 'num_samples', 'epochs'
        ]
        for key in expected_keys:
            self.assertIn(key, metrics)
            
        self.assertEqual(metrics['client_id'], 1)
        self.assertEqual(metrics['epochs'], 2)
        self.assertEqual(num_samples, 50)
        
        # 验证轮次增加
        self.assertEqual(client.current_round, 1)
        
    def test_training_error_handling(self):
        """测试训练过程中的错误处理"""
        client = ConcreteClient(
            client_id=1,
            train_data_loader=self.train_loader,
            config=self.config
        )
        
        # 模拟训练失败的情况
        with patch.object(client, 'local_train', side_effect=Exception("Training failed")):
            global_model_state = {'param': torch.randn(3, 3)}
            
            with self.assertRaises(ClientTrainingError):
                client.local_train_and_evaluate(global_model_state, num_epochs=1)
                
            # 验证状态变为错误
            self.assertEqual(client.status, ComponentStatus.ERROR)
            
    def test_client_reset(self):
        """测试客户端重置功能"""
        client = ConcreteClient(
            client_id=1,
            train_data_loader=self.train_loader,
            config=self.config
        )
        
        # 执行一次训练
        global_model_state = {'param': torch.randn(3, 3)}
        client.local_train_and_evaluate(global_model_state, num_epochs=1)
        
        # 验证状态已改变
        self.assertEqual(client.current_round, 1)
        
        # 重置客户端
        client.reset()
        
        # 验证重置后状态
        self.assertEqual(client.current_round, 0)
        self.assertEqual(client.status, ComponentStatus.READY)
        
    def test_training_history_tracking(self):
        """测试训练历史跟踪"""
        client = ConcreteClient(
            client_id=1,
            train_data_loader=self.train_loader,
            config=self.config
        )
        
        # 初始时历史为空
        history = client.get_training_history()
        self.assertEqual(len(history), 0)
        
        # 执行多次训练
        global_model_state = {'param': torch.randn(3, 3)}
        
        for _ in range(3):
            client.local_train_and_evaluate(global_model_state, num_epochs=1)
            
        # 验证轮次正确
        self.assertEqual(client.current_round, 3)
        
    def test_config_validation_edge_cases(self):
        """测试配置验证边界情况"""
        # 测试空配置
        with self.assertRaises(ClientConfigurationError):
            ConcreteClient(
                client_id=1,
                train_data_loader=self.train_loader,
                config={}
            )
            
        # 测试无效优化器配置
        invalid_config = {
            'optimizer': "not_a_dict",  # 应该是字典
            'device': 'cpu'
        }
        with self.assertRaises(ClientConfigurationError):
            ConcreteClient(
                client_id=1,
                train_data_loader=self.train_loader,
                config=invalid_config
            )
            
    def test_device_management(self):
        """测试设备管理"""
        # 默认设备
        client = ConcreteClient(
            client_id=1,
            train_data_loader=self.train_loader,
            config=self.config
        )
        self.assertEqual(client.device, 'cpu')
        
        # 指定设备
        client_gpu = ConcreteClient(
            client_id=2,
            train_data_loader=self.train_loader,
            config=self.config,
            device='cuda'
        )
        self.assertEqual(client_gpu.device, 'cuda')
        
    def test_data_loader_attributes_preservation(self):
        """测试数据加载器属性保持"""
        # 创建带特定属性的数据加载器
        original_loader = DataLoader(
            self.dataset, 
            batch_size=8,
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )
        
        client = ConcreteClient(
            client_id=1,
            train_data_loader=original_loader,
            config=self.config
        )
        
        # 验证属性被正确保持
        self.assertEqual(client.train_loader.batch_size, 8)
        # 注意：由于数据分割，实际数据集大小会改变
        
    def test_string_representation(self):
        """测试字符串表示"""
        client = ConcreteClient(
            client_id=5,
            train_data_loader=self.train_loader,
            config=self.config
        )
        
        str_repr = str(client)
        self.assertIn("Client", str_repr)
        self.assertIn("id=5", str_repr)
        self.assertIn("samples=", str_repr)


class TestAbstractClientAbstractMethods(unittest.TestCase):
    """测试抽象方法强制实现"""
    
    def test_abstract_methods_enforcement(self):
        """测试抽象方法必须被实现"""
        
        # 尝试直接实例化抽象类应该失败
        dataset = TensorDataset(torch.randn(10, 5), torch.randint(0, 2, (10,)))
        train_loader = DataLoader(dataset, batch_size=5)
        config = {
            'optimizer': {'name': 'adam', 'lr': 0.001},
            'device': 'cpu'
        }
        
        with self.assertRaises(TypeError):
            AbstractClient(
                client_id=1,
                train_data_loader=train_loader,
                config=config
            )


if __name__ == '__main__':
    unittest.main()