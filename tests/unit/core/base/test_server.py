"""
AbstractServer 抽象类全面单元测试
"""

import unittest
from unittest.mock import Mock, MagicMock, patch, call
import torch
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime

from src.core.base.server import AbstractServer
from src.core.base.client import AbstractClient
from src.core.base.component import ComponentStatus
from src.core.exceptions import (
    ServerConfigurationError,
    ServerOperationError,
    ConfigValidationError
)


class MockClient(AbstractClient):
    """用于测试的Mock客户端"""
    
    def __init__(self, client_id, num_samples=100):
        dataset = TensorDataset(torch.randn(num_samples, 10), torch.randint(0, 2, (num_samples,)))
        train_loader = DataLoader(dataset, batch_size=10)
        config = {
            'optimizer': {'name': 'adam', 'lr': 0.001},
            'device': 'cpu'
        }
        super().__init__(client_id, train_loader, config, 'cpu')
        self._num_samples = num_samples
        
    def _validate_config(self):
        pass
        
    def receive_global_model(self, global_model_state):
        self._received_model = global_model_state
        
    def local_train(self, num_epochs):
        model_state = {'layer1.weight': torch.randn(5, 10)}
        metrics = {'train_acc': 0.8, 'train_loss': 0.3}
        return model_state, metrics, self._num_samples
        
    def local_evaluate(self, model_state=None):
        return {'test_acc': 0.75, 'test_loss': 0.4}


class MockModel(torch.nn.Module):
    """用于测试的Mock模型"""
    
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(10, 5)
        
    def forward(self, x):
        return self.layer1(x)


class ConcreteServer(AbstractServer):
    """用于测试的具体服务器实现"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._model_initialized = False
        
    def initialize_global_model(self):
        """初始化全局模型"""
        self._global_model = self._model_constructor()
        self._global_model.to(self._device)
        self._model_initialized = True
        
    def select_clients(self, round_number):
        """选择所有客户端参与训练"""
        return self._clients.copy()
        
    def aggregate_models(self, client_models, client_weights):
        """简单的FedAvg聚合"""
        if not client_models:
            raise ServerOperationError("No client models to aggregate")
            
        total_weight = sum(client_weights)
        aggregated_state = {}
        
        # 对每个参数进行加权平均
        for key in client_models[0].keys():
            aggregated_param = torch.zeros_like(client_models[0][key])
            for model, weight in zip(client_models, client_weights):
                aggregated_param += model[key] * (weight / total_weight)
            aggregated_state[key] = aggregated_param
            
        return aggregated_state
        
    def save_checkpoint(self, round_number, metrics):
        """保存检查点（Mock实现）"""
        pass  # 简单实现，不实际保存文件


class TestAbstractServer(unittest.TestCase):
    
    def setUp(self):
        """测试前设置"""
        self.config = {
            'federated': {
                'num_rounds': 5,
                'local_epochs': 2,
                'batch_size': 32
            }
        }
        
        # 创建Mock客户端
        self.clients = [
            MockClient(client_id=i, num_samples=100 + i*10) 
            for i in range(3)
        ]
        
        # 模型构造函数
        self.model_constructor = MockModel
        
    def test_server_initialization_success(self):
        """测试服务器成功初始化"""
        server = ConcreteServer(
            model_constructor=self.model_constructor,
            clients=self.clients,
            config=self.config,
            device='cpu'
        )
        
        # 验证基本属性
        self.assertEqual(server.component_id, 'federated_server')
        self.assertEqual(server.device, 'cpu')
        self.assertEqual(server.num_clients, 3)
        self.assertEqual(len(server.clients), 3)
        self.assertEqual(server.current_round, 0)
        self.assertEqual(server.total_rounds, 0)
        self.assertFalse(server.is_training)
        self.assertIsNone(server.training_duration)
        
    def test_server_initialization_no_clients(self):
        """测试无客户端时初始化失败"""
        with self.assertRaises(ServerConfigurationError):
            ConcreteServer(
                model_constructor=self.model_constructor,
                clients=[],
                config=self.config
            )
            
    def test_server_initialization_invalid_config(self):
        """测试无效配置时初始化失败"""
        invalid_config = {}  # 缺少federated配置
        
        with self.assertRaises(ServerConfigurationError):
            ConcreteServer(
                model_constructor=self.model_constructor,
                clients=self.clients,
                config=invalid_config
            )
            
        # 测试无效的轮数配置
        invalid_config2 = {
            'federated': {
                'num_rounds': -1  # 无效轮数
            }
        }
        with self.assertRaises(ServerConfigurationError):
            ConcreteServer(
                model_constructor=self.model_constructor,
                clients=self.clients,
                config=invalid_config2
            )
            
    def test_client_management(self):
        """测试客户端管理功能"""
        server = ConcreteServer(
            model_constructor=self.model_constructor,
            clients=self.clients[:2],  # 初始2个客户端
            config=self.config
        )
        
        self.assertEqual(server.num_clients, 2)
        
        # 添加客户端
        new_client = MockClient(client_id=10, num_samples=150)
        server.add_client(new_client)
        
        self.assertEqual(server.num_clients, 3)
        self.assertIn(new_client, server.clients)
        
        # 尝试添加重复客户端
        server.add_client(new_client)  # 应该有警告但不会重复添加
        self.assertEqual(server.num_clients, 3)
        
        # 移除客户端
        success = server.remove_client(10)
        self.assertTrue(success)
        self.assertEqual(server.num_clients, 2)
        
        # 尝试移除不存在的客户端
        success = server.remove_client(999)
        self.assertFalse(success)
        self.assertEqual(server.num_clients, 2)
        
    def test_global_model_initialization(self):
        """测试全局模型初始化"""
        server = ConcreteServer(
            model_constructor=self.model_constructor,
            clients=self.clients,
            config=self.config
        )
        
        # 初始时模型未初始化
        self.assertIsNone(server.global_model)
        
        # 初始化模型
        server.initialize_global_model()
        
        # 验证模型已初始化
        self.assertIsNotNone(server.global_model)
        self.assertIsInstance(server.global_model, MockModel)
        
    def test_client_selection(self):
        """测试客户端选择"""
        server = ConcreteServer(
            model_constructor=self.model_constructor,
            clients=self.clients,
            config=self.config
        )
        
        # 默认实现应该选择所有客户端
        selected = server.select_clients(1)
        
        self.assertEqual(len(selected), len(self.clients))
        for client in self.clients:
            self.assertIn(client, selected)
            
    def test_model_aggregation(self):
        """测试模型聚合"""
        server = ConcreteServer(
            model_constructor=self.model_constructor,
            clients=self.clients,
            config=self.config
        )
        
        # 创建模拟的客户端模型
        client_models = [
            {'layer1.weight': torch.ones(5, 10) * (i + 1)}
            for i in range(3)
        ]
        client_weights = [100, 200, 100]  # 总权重400
        
        # 执行聚合
        aggregated = server.aggregate_models(client_models, client_weights)
        
        # 验证聚合结果
        self.assertIn('layer1.weight', aggregated)
        expected = (torch.ones(5, 10) * 1 * 100/400 + 
                   torch.ones(5, 10) * 2 * 200/400 + 
                   torch.ones(5, 10) * 3 * 100/400)
        torch.testing.assert_close(aggregated['layer1.weight'], expected)
        
    def test_aggregation_empty_models(self):
        """测试空模型聚合失败"""
        server = ConcreteServer(
            model_constructor=self.model_constructor,
            clients=self.clients,
            config=self.config
        )
        
        with self.assertRaises(ServerOperationError):
            server.aggregate_models([], [])
            
    def test_federated_learning_execution(self):
        """测试完整的联邦学习执行流程"""
        server = ConcreteServer(
            model_constructor=self.model_constructor,
            clients=self.clients,
            config=self.config
        )
        
        # 执行联邦学习
        result = server.run_federated_learning(num_rounds=2, local_epochs=1)
        
        # 验证结果
        self.assertIsInstance(result, dict)
        self.assertEqual(result['total_rounds'], 2)
        self.assertEqual(result['completed_rounds'], 2)
        self.assertEqual(result['num_clients'], 3)
        self.assertIsNotNone(result['training_duration'])
        self.assertGreater(result['training_duration'], 0)
        
        # 验证服务器状态
        self.assertEqual(server.current_round, 2)
        self.assertEqual(server.total_rounds, 2)
        self.assertEqual(server.status, ComponentStatus.READY)
        self.assertFalse(server.is_training)
        
        # 验证轮次历史
        history = server.get_round_history()
        self.assertEqual(len(history), 2)
        for i, round_result in enumerate(history, 1):
            self.assertEqual(round_result['round'], i)
            self.assertEqual(round_result['num_clients'], 3)
            
    def test_training_error_handling(self):
        """测试训练过程错误处理"""
        server = ConcreteServer(
            model_constructor=self.model_constructor,
            clients=self.clients,
            config=self.config
        )
        
        # 模拟初始化失败
        with patch.object(server, 'initialize_global_model', 
                         side_effect=Exception("Initialization failed")):
            with self.assertRaises(ServerOperationError):
                server.run_federated_learning(num_rounds=1)
                
            # 验证状态变为错误
            self.assertEqual(server.status, ComponentStatus.ERROR)
            
    def test_round_execution_details(self):
        """测试单轮执行细节"""
        server = ConcreteServer(
            model_constructor=self.model_constructor,
            clients=self.clients,
            config=self.config
        )
        
        # 初始化模型
        server.initialize_global_model()
        
        # 执行单轮
        with patch.object(server, 'save_checkpoint') as mock_save:
            round_result = server._execute_round(1, local_epochs=1)
            
            # 验证轮次结果
            self.assertEqual(round_result['round'], 1)
            self.assertEqual(round_result['num_clients'], 3)
            self.assertEqual(round_result['total_samples'], 330)  # 100+110+120
            
            # 验证保存检查点被调用
            mock_save.assert_called_once()
            
    def test_metrics_computation(self):
        """测试指标计算"""
        server = ConcreteServer(
            model_constructor=self.model_constructor,
            clients=self.clients,
            config=self.config
        )
        
        # 测试轮次指标计算
        clients = self.clients
        client_weights = [100, 110, 120]
        
        metrics = server._compute_round_metrics(1, clients, client_weights)
        
        self.assertEqual(metrics['round'], 1)
        self.assertEqual(metrics['num_clients'], 3)
        self.assertEqual(metrics['total_samples'], 330)
        self.assertEqual(metrics['avg_samples_per_client'], 110.0)
        self.assertIn('timestamp', metrics)
        
    def test_final_results_compilation(self):
        """测试最终结果编译"""
        server = ConcreteServer(
            model_constructor=self.model_constructor,
            clients=self.clients,
            config=self.config
        )
        
        # 模拟轮次历史
        server._round_history = [
            {'round': 1, 'metric': 'value1'},
            {'round': 2, 'metric': 'value2'}
        ]
        server._total_rounds = 2
        server._training_start_time = datetime.now()
        server._training_end_time = datetime.now()
        
        result = server._compile_final_results()
        
        self.assertEqual(result['total_rounds'], 2)
        self.assertEqual(result['completed_rounds'], 2)
        self.assertEqual(result['num_clients'], 3)
        self.assertEqual(result['final_metrics'], {'round': 2, 'metric': 'value2'})
        self.assertIsNotNone(result['training_duration'])
        
    def test_property_accessors(self):
        """测试属性访问器"""
        server = ConcreteServer(
            model_constructor=self.model_constructor,
            clients=self.clients,
            config=self.config
        )
        
        # 测试基本属性
        self.assertEqual(server.model_constructor, self.model_constructor)
        self.assertEqual(len(server.clients), 3)
        self.assertEqual(server.device, 'cpu')
        self.assertIsNone(server.global_model)
        self.assertEqual(server.current_round, 0)
        self.assertEqual(server.total_rounds, 0)
        self.assertEqual(server.num_clients, 3)
        self.assertFalse(server.is_training)
        
        # 测试训练状态
        server._set_status(ComponentStatus.RUNNING)
        self.assertTrue(server.is_training)
        
    def test_string_representation(self):
        """测试字符串表示"""
        server = ConcreteServer(
            model_constructor=self.model_constructor,
            clients=self.clients,
            config=self.config
        )
        
        str_repr = str(server)
        self.assertIn("Server", str_repr)
        self.assertIn("clients=3", str_repr)
        self.assertIn("round=0/0", str_repr)


class TestAbstractServerAbstractMethods(unittest.TestCase):
    """测试抽象方法强制实现"""
    
    def test_abstract_methods_enforcement(self):
        """测试抽象方法必须被实现"""
        clients = [MockClient(1)]
        config = {'federated': {'num_rounds': 1}}
        
        # 尝试直接实例化抽象类应该失败
        with self.assertRaises(TypeError):
            AbstractServer(
                model_constructor=MockModel,
                clients=clients,
                config=config
            )


if __name__ == '__main__':
    unittest.main()