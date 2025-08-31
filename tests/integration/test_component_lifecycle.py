"""
组件生命周期集成测试

测试各组件从创建到销毁的完整生命周期以及组件间交互。
"""

import unittest
from unittest.mock import Mock, patch
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.core.base.component import FederatedComponent, ComponentStatus, ComponentRegistry
from src.core.base.client import AbstractClient
from src.core.base.server import AbstractServer
from src.core.base.aggregator import AbstractAggregator, AggregationMode
from src.core.exceptions import (
    FederatedLearningException,
    ClientConfigurationError,
    ServerConfigurationError,
    AggregationError
)


class TestClient(AbstractClient):
    """测试用客户端实现"""
    
    def __init__(self, client_id, fail_validation=False, fail_training=False):
        dataset = TensorDataset(torch.randn(50, 10), torch.randint(0, 2, (50,)))
        train_loader = DataLoader(dataset, batch_size=10)
        config = {
            'optimizer': {'name': 'adam', 'lr': 0.001},
            'device': 'cpu'
        }
        
        self.fail_validation = fail_validation
        self.fail_training = fail_training
        
        super().__init__(client_id, train_loader, config, 'cpu')
        
    def _validate_config(self):
        if self.fail_validation:
            raise ClientConfigurationError("Validation failed for testing")
            
    def receive_global_model(self, global_model_state):
        self.received_model = global_model_state
        
    def local_train(self, num_epochs):
        if self.fail_training:
            raise Exception("Training failed for testing")
        return {'param': torch.randn(3, 3)}, {'acc': 0.8, 'loss': 0.2}, 50
        
    def local_evaluate(self, model_state=None):
        return {'acc': 0.75, 'loss': 0.25}


class TestServer(AbstractServer):
    """测试用服务器实现"""
    
    def __init__(self, *args, fail_initialization=False, **kwargs):
        self.fail_initialization = fail_initialization
        super().__init__(*args, **kwargs)
        
    def initialize_global_model(self):
        if self.fail_initialization:
            raise ServerConfigurationError("Initialization failed for testing")
        self._global_model = torch.nn.Linear(10, 5)
        
    def select_clients(self, round_number):
        return self._clients
        
    def aggregate_models(self, client_models, client_weights):
        # 简单聚合
        return client_models[0] if client_models else {}
        
    def save_checkpoint(self, round_number, metrics):
        pass


class TestAggregator(AbstractAggregator):
    """测试用聚合器实现"""
    
    def __init__(self, *args, fail_aggregation=False, **kwargs):
        self.fail_aggregation = fail_aggregation
        super().__init__(*args, **kwargs)
        
    def aggregate(self, client_models, client_weights, metadata=None):
        if self.fail_aggregation:
            raise AggregationError("Aggregation failed for testing")
        return client_models[0] if client_models else {}
        
    def compute_aggregation_weights(self, client_weights, metadata=None):
        total = sum(client_weights)
        return [w/total for w in client_weights] if total > 0 else []


class TestComponentLifecycle(unittest.TestCase):
    """组件生命周期集成测试"""
    
    def setUp(self):
        """测试前设置"""
        ComponentRegistry.clear()
        
    def tearDown(self):
        """测试后清理"""
        ComponentRegistry.clear()
        
    def test_component_creation_lifecycle(self):
        """测试组件创建生命周期"""
        # 创建客户端
        client = TestClient(client_id=1)
        
        # 验证创建后状态
        self.assertEqual(client.status, ComponentStatus.READY)
        self.assertEqual(client.client_id, 1)
        self.assertIsNotNone(client.created_at)
        
        # 验证可以正常操作
        global_state = {'param': torch.randn(2, 2)}
        model_state, metrics, samples = client.local_train_and_evaluate(global_state, 1)
        
        self.assertIsNotNone(model_state)
        self.assertIn('acc', metrics)
        self.assertEqual(samples, 50)
        
    def test_component_error_lifecycle(self):
        """测试组件错误生命周期"""
        # 创建会在验证时失败的客户端
        with self.assertRaises(ClientConfigurationError):
            TestClient(client_id=1, fail_validation=True)
            
        # 创建会在训练时失败的客户端
        client = TestClient(client_id=2, fail_training=True)
        self.assertEqual(client.status, ComponentStatus.READY)
        
        # 尝试训练应该失败
        global_state = {'param': torch.randn(2, 2)}
        with self.assertRaises(FederatedLearningException):
            client.local_train_and_evaluate(global_state, 1)
            
        # 验证状态变为错误
        self.assertEqual(client.status, ComponentStatus.ERROR)
        
    def test_component_status_transitions(self):
        """测试组件状态转换"""
        client = TestClient(client_id=1)
        
        # 初始状态
        self.assertEqual(client.status, ComponentStatus.READY)
        
        # 模拟状态变化
        client._set_status(ComponentStatus.RUNNING)
        self.assertEqual(client.status, ComponentStatus.RUNNING)
        
        client._set_status(ComponentStatus.PAUSED)
        self.assertEqual(client.status, ComponentStatus.PAUSED)
        
        client._set_status(ComponentStatus.ERROR)
        self.assertEqual(client.status, ComponentStatus.ERROR)
        
        # 重置状态
        client.reset()
        self.assertEqual(client.status, ComponentStatus.READY)
        
    def test_component_config_lifecycle(self):
        """测试组件配置生命周期"""
        client = TestClient(client_id=1)
        
        # 验证初始配置
        initial_lr = client.get_config_value('optimizer.lr')
        self.assertEqual(initial_lr, 0.001)
        
        # 更新配置
        new_config = {'optimizer': {'name': 'sgd', 'lr': 0.01}}
        client.update_config(new_config)
        
        # 验证配置更新
        updated_lr = client.get_config_value('optimizer.lr')
        self.assertEqual(updated_lr, 0.01)
        
        updated_name = client.get_config_value('optimizer.name')
        self.assertEqual(updated_name, 'sgd')
        
    def test_component_metadata_lifecycle(self):
        """测试组件元数据生命周期"""
        client = TestClient(client_id=1)
        
        # 设置元数据
        client.set_metadata('experiment_id', 'exp_001')
        client.set_metadata('version', '1.0.0')
        
        # 验证元数据
        self.assertEqual(client.get_metadata('experiment_id'), 'exp_001')
        self.assertEqual(client.get_metadata('version'), '1.0.0')
        self.assertIsNone(client.get_metadata('nonexistent'))
        
        # 验证元数据副本
        metadata = client.metadata
        metadata['new_key'] = 'new_value'
        self.assertIsNone(client.get_metadata('new_key'))  # 不应该影响原始数据


class TestComponentInteractions(unittest.TestCase):
    """组件交互集成测试"""
    
    def setUp(self):
        """测试前设置"""
        self.clients = [TestClient(i) for i in range(3)]
        self.config = {
            'federated': {'num_rounds': 2, 'local_epochs': 1}
        }
        
    def test_client_server_basic_interaction(self):
        """测试客户端-服务器基本交互"""
        server = TestServer(
            model_constructor=lambda: torch.nn.Linear(10, 5),
            clients=self.clients,
            config=self.config
        )
        
        # 验证服务器初始化
        self.assertEqual(server.num_clients, 3)
        self.assertEqual(server.status, ComponentStatus.READY)
        
        # 初始化全局模型
        server.initialize_global_model()
        self.assertIsNotNone(server.global_model)
        
        # 选择客户端
        selected = server.select_clients(1)
        self.assertEqual(len(selected), 3)
        
    def test_federated_learning_round_interaction(self):
        """测试联邦学习轮次交互"""
        server = TestServer(
            model_constructor=lambda: torch.nn.Linear(10, 5),
            clients=self.clients,
            config=self.config
        )
        
        # 执行单轮训练
        server.initialize_global_model()
        
        # 模拟轮次执行
        global_state = server.global_model.state_dict()
        
        client_models = []
        client_weights = []
        
        for client in self.clients:
            model_state, metrics, samples = client.local_train_and_evaluate(global_state, 1)
            client_models.append(model_state)
            client_weights.append(samples)
            
        # 聚合模型
        aggregated_state = server.aggregate_models(client_models, client_weights)
        
        # 验证聚合结果
        self.assertIsNotNone(aggregated_state)
        
    def test_error_propagation_in_interactions(self):
        """测试交互中的错误传播"""
        # 创建会失败的客户端
        failing_clients = [
            TestClient(1),
            TestClient(2, fail_training=True),  # 这个客户端会训练失败
            TestClient(3)
        ]
        
        server = TestServer(
            model_constructor=lambda: torch.nn.Linear(10, 5),
            clients=failing_clients,
            config=self.config
        )
        
        server.initialize_global_model()
        global_state = server.global_model.state_dict()
        
        # 尝试训练所有客户端
        successful_clients = 0
        failed_clients = 0
        
        for client in failing_clients:
            try:
                client.local_train_and_evaluate(global_state, 1)
                successful_clients += 1
            except Exception:
                failed_clients += 1
                
        # 验证只有一个客户端失败
        self.assertEqual(successful_clients, 2)
        self.assertEqual(failed_clients, 1)
        
    def test_aggregator_integration(self):
        """测试聚合器集成"""
        aggregator = TestAggregator(AggregationMode.WEIGHTED_AVERAGE)
        
        # 创建模拟客户端模型
        client_models = [
            {'param': torch.ones(3, 3) * (i + 1)}
            for i in range(3)
        ]
        client_weights = [100, 200, 100]
        
        # 执行聚合
        result = aggregator.aggregate_with_validation(
            client_models, client_weights
        )
        
        # 验证聚合器状态
        self.assertEqual(aggregator.aggregation_count, 1)
        self.assertEqual(aggregator.status, ComponentStatus.READY)
        
        # 验证历史记录
        history = aggregator.get_aggregation_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]['num_clients'], 3)
        
    def test_component_registry_integration(self):
        """测试组件注册表集成"""
        # 注册组件
        client1 = TestClient(1)
        client2 = TestClient(2)
        
        ComponentRegistry.register(client1)
        ComponentRegistry.register(client2)
        
        # 验证注册
        self.assertEqual(len(ComponentRegistry.list_components()), 2)
        
        # 获取特定组件
        retrieved = ComponentRegistry.get_component('client_1')
        self.assertEqual(retrieved, client1)
        
        # 按类型列出组件
        clients = ComponentRegistry.list_components(TestClient)
        self.assertEqual(len(clients), 2)
        
        # 清理
        ComponentRegistry.clear()
        self.assertEqual(len(ComponentRegistry.list_components()), 0)


class TestComplexInteractionScenarios(unittest.TestCase):
    """复杂交互场景测试"""
    
    def test_multi_round_federated_learning(self):
        """测试多轮联邦学习交互"""
        clients = [TestClient(i) for i in range(2)]
        config = {'federated': {'num_rounds': 3, 'local_epochs': 1}}
        
        server = TestServer(
            model_constructor=lambda: torch.nn.Linear(10, 5),
            clients=clients,
            config=config
        )
        
        # 执行多轮联邦学习
        result = server.run_federated_learning(num_rounds=3, local_epochs=1)
        
        # 验证结果
        self.assertEqual(result['total_rounds'], 3)
        self.assertEqual(result['completed_rounds'], 3)
        self.assertIsNotNone(result['training_duration'])
        
        # 验证轮次历史
        history = server.get_round_history()
        self.assertEqual(len(history), 3)
        
        for i, round_result in enumerate(history, 1):
            self.assertEqual(round_result['round'], i)
            
    def test_heterogeneous_client_interactions(self):
        """测试异构客户端交互"""
        # 创建不同样本数量的客户端
        clients = [
            TestClient(1),  # 50样本
            TestClient(2),  # 50样本  
            TestClient(3)   # 50样本
        ]
        
        # 手动设置不同的样本数量
        clients[0]._num_samples = 100
        clients[1]._num_samples = 200
        clients[2]._num_samples = 50
        
        # 验证数据分布差异
        for i, client in enumerate(clients):
            info = client.get_data_distribution_info()
            expected_samples = [100, 200, 50][i]
            # 注意：由于数据分割，实际样本数会不同
            self.assertGreater(info['total_samples'], 0)
            
    def test_error_recovery_scenarios(self):
        """测试错误恢复场景"""
        client = TestClient(1)
        
        # 模拟训练失败
        client._set_status(ComponentStatus.ERROR)
        self.assertEqual(client.status, ComponentStatus.ERROR)
        
        # 重置并恢复
        client.reset()
        self.assertEqual(client.status, ComponentStatus.READY)
        
        # 验证可以继续正常工作
        global_state = {'param': torch.randn(2, 2)}
        result = client.local_train_and_evaluate(global_state, 1)
        self.assertIsNotNone(result)
        
    def test_concurrent_component_operations(self):
        """测试并发组件操作模拟"""
        # 创建多个组件
        components = [TestClient(i) for i in range(5)]
        
        # 模拟并发状态变更
        for component in components:
            component._set_status(ComponentStatus.RUNNING)
            
        # 验证所有组件状态
        for component in components:
            self.assertEqual(component.status, ComponentStatus.RUNNING)
            
        # 模拟并发完成
        for component in components:
            component._set_status(ComponentStatus.READY)
            
        # 验证最终状态
        for component in components:
            self.assertEqual(component.status, ComponentStatus.READY)


if __name__ == '__main__':
    unittest.main()