"""
策略接口全面单元测试
"""

import unittest
from unittest.mock import Mock, MagicMock
from typing import Dict, Any, List
import torch

from src.core.interfaces.strategy import (
    StrategyInterface,
    AggregationStrategyInterface,
    ClientSelectionStrategyInterface,
    DataPartitioningStrategyInterface,
    ModelUpdateStrategyInterface,
    PrivacyStrategyInterface,
    CommunicationStrategyInterface,
    strategy_register
)
from src.core.exceptions import StrategyError, ComponentCreationError


class ConcreteAggregationStrategy(AggregationStrategyInterface):
    """用于测试的具体聚合策略实现"""
    
    def aggregate(self, client_models: List[Dict[str, torch.Tensor]], 
                 client_weights: List[float], metadata: Dict[str, Any] = None) -> Dict[str, torch.Tensor]:
        """简单平均聚合"""
        if not client_models:
            raise StrategyError("No models to aggregate")
            
        aggregated = {}
        for key in client_models[0].keys():
            aggregated[key] = torch.mean(torch.stack([model[key] for model in client_models]), dim=0)
        return aggregated
        
    def compute_weights(self, client_info: List[Dict[str, Any]], 
                       round_info: Dict[str, Any] = None) -> List[float]:
        """计算客户端权重"""
        return [1.0] * len(client_info)  # 均等权重


class ConcreteClientSelectionStrategy(ClientSelectionStrategyInterface):
    """用于测试的具体客户端选择策略实现"""
    
    def __init__(self, selection_rate=0.5):
        self.selection_rate = selection_rate
        
    def select_clients(self, available_clients: List[Any], 
                      round_info: Dict[str, Any] = None) -> List[Any]:
        """选择前一半客户端"""
        num_select = max(1, int(len(available_clients) * self.selection_rate))
        return available_clients[:num_select]
        
    def get_selection_criteria(self) -> Dict[str, Any]:
        """获取选择标准"""
        return {'selection_rate': self.selection_rate, 'strategy': 'first_n'}


class ConcreteDataPartitioningStrategy(DataPartitioningStrategyInterface):
    """用于测试的具体数据分区策略实现"""
    
    def partition_data(self, dataset: Any, num_clients: int, 
                      partition_config: Dict[str, Any] = None) -> List[Any]:
        """简单数据分区"""
        if num_clients <= 0:
            raise StrategyError("Number of clients must be positive")
            
        # 模拟数据分区
        partitions = []
        for i in range(num_clients):
            partitions.append(f"partition_{i}")
        return partitions
        
    def get_partition_info(self, partitions: List[Any]) -> Dict[str, Any]:
        """获取分区信息"""
        return {
            'num_partitions': len(partitions),
            'partition_type': 'simulated',
            'balance': 'equal'
        }


class ConcreteModelUpdateStrategy(ModelUpdateStrategyInterface):
    """用于测试的具体模型更新策略实现"""
    
    def compute_update(self, old_model: Dict[str, torch.Tensor], 
                      new_model: Dict[str, torch.Tensor], 
                      config: Dict[str, Any] = None) -> Dict[str, torch.Tensor]:
        """计算模型更新"""
        update = {}
        for key in old_model.keys():
            update[key] = new_model[key] - old_model[key]
        return update
        
    def apply_update(self, model: Dict[str, torch.Tensor], 
                    update: Dict[str, torch.Tensor], 
                    config: Dict[str, Any] = None) -> Dict[str, torch.Tensor]:
        """应用模型更新"""
        updated_model = {}
        learning_rate = config.get('learning_rate', 1.0) if config else 1.0
        
        for key in model.keys():
            updated_model[key] = model[key] + learning_rate * update[key]
        return updated_model


class TestStrategyInterface(unittest.TestCase):
    
    def test_strategy_interface_abstract_methods(self):
        """测试策略接口抽象方法强制实现"""
        
        # 尝试直接实例化抽象类应该失败
        with self.assertRaises(TypeError):
            StrategyInterface()


class TestAggregationStrategyInterface(unittest.TestCase):
    
    def setUp(self):
        """测试前设置"""
        self.strategy = ConcreteAggregationStrategy()
        
        # 创建测试模型
        self.client_models = [
            {'param1': torch.ones(2, 2) * (i + 1), 'param2': torch.ones(3) * (i + 1)}
            for i in range(3)
        ]
        self.client_weights = [1.0, 2.0, 1.0]
        self.client_info = [
            {'client_id': i, 'samples': 100 + i * 50}
            for i in range(3)
        ]
        
    def test_aggregation_success(self):
        """测试聚合成功"""
        result = self.strategy.aggregate(self.client_models, self.client_weights)
        
        # 验证结果结构
        self.assertIn('param1', result)
        self.assertIn('param2', result)
        
        # 验证聚合结果（简单平均）
        expected_param1 = torch.mean(torch.stack([
            torch.ones(2, 2) * 1,
            torch.ones(2, 2) * 2, 
            torch.ones(2, 2) * 3
        ]), dim=0)
        torch.testing.assert_close(result['param1'], expected_param1)
        
    def test_aggregation_empty_models(self):
        """测试空模型列表聚合"""
        with self.assertRaises(StrategyError):
            self.strategy.aggregate([], [])
            
    def test_compute_weights_success(self):
        """测试权重计算成功"""
        weights = self.strategy.compute_weights(self.client_info)
        
        # 验证权重数量
        self.assertEqual(len(weights), len(self.client_info))
        
        # 验证均等权重
        for weight in weights:
            self.assertEqual(weight, 1.0)
            
    def test_abstract_methods_enforcement(self):
        """测试抽象方法强制实现"""
        with self.assertRaises(TypeError):
            AggregationStrategyInterface()


class TestClientSelectionStrategyInterface(unittest.TestCase):
    
    def setUp(self):
        """测试前设置"""
        self.strategy = ConcreteClientSelectionStrategy(selection_rate=0.6)
        self.available_clients = [f"client_{i}" for i in range(10)]
        
    def test_client_selection_success(self):
        """测试客户端选择成功"""
        selected = self.strategy.select_clients(self.available_clients)
        
        # 验证选择数量
        expected_count = int(len(self.available_clients) * 0.6)
        self.assertEqual(len(selected), expected_count)
        
        # 验证选择的是前n个客户端
        for i, client in enumerate(selected):
            self.assertEqual(client, f"client_{i}")
            
    def test_client_selection_empty_list(self):
        """测试空客户端列表选择"""
        selected = self.strategy.select_clients([])
        self.assertEqual(len(selected), 0)
        
    def test_client_selection_single_client(self):
        """测试单客户端选择"""
        single_client = ["client_0"]
        selected = self.strategy.select_clients(single_client)
        
        # 至少选择一个客户端
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0], "client_0")
        
    def test_get_selection_criteria(self):
        """测试获取选择标准"""
        criteria = self.strategy.get_selection_criteria()
        
        self.assertEqual(criteria['selection_rate'], 0.6)
        self.assertEqual(criteria['strategy'], 'first_n')
        
    def test_abstract_methods_enforcement(self):
        """测试抽象方法强制实现"""
        with self.assertRaises(TypeError):
            ClientSelectionStrategyInterface()


class TestDataPartitioningStrategyInterface(unittest.TestCase):
    
    def setUp(self):
        """测试前设置"""
        self.strategy = ConcreteDataPartitioningStrategy()
        self.dataset = "mock_dataset"
        
    def test_data_partitioning_success(self):
        """测试数据分区成功"""
        num_clients = 5
        partitions = self.strategy.partition_data(self.dataset, num_clients)
        
        # 验证分区数量
        self.assertEqual(len(partitions), num_clients)
        
        # 验证分区内容
        for i, partition in enumerate(partitions):
            self.assertEqual(partition, f"partition_{i}")
            
    def test_data_partitioning_invalid_clients(self):
        """测试无效客户端数量分区"""
        with self.assertRaises(StrategyError):
            self.strategy.partition_data(self.dataset, 0)
            
        with self.assertRaises(StrategyError):
            self.strategy.partition_data(self.dataset, -1)
            
    def test_get_partition_info(self):
        """测试获取分区信息"""
        partitions = self.strategy.partition_data(self.dataset, 3)
        info = self.strategy.get_partition_info(partitions)
        
        self.assertEqual(info['num_partitions'], 3)
        self.assertEqual(info['partition_type'], 'simulated')
        self.assertEqual(info['balance'], 'equal')
        
    def test_abstract_methods_enforcement(self):
        """测试抽象方法强制实现"""
        with self.assertRaises(TypeError):
            DataPartitioningStrategyInterface()


class TestModelUpdateStrategyInterface(unittest.TestCase):
    
    def setUp(self):
        """测试前设置"""
        self.strategy = ConcreteModelUpdateStrategy()
        self.old_model = {
            'param1': torch.ones(2, 2),
            'param2': torch.ones(3) * 2
        }
        self.new_model = {
            'param1': torch.ones(2, 2) * 2,
            'param2': torch.ones(3) * 3
        }
        
    def test_compute_update_success(self):
        """测试计算模型更新成功"""
        update = self.strategy.compute_update(self.old_model, self.new_model)
        
        # 验证更新结构
        self.assertIn('param1', update)
        self.assertIn('param2', update)
        
        # 验证更新值
        expected_param1 = torch.ones(2, 2)  # 2 - 1
        expected_param2 = torch.ones(3)     # 3 - 2
        
        torch.testing.assert_close(update['param1'], expected_param1)
        torch.testing.assert_close(update['param2'], expected_param2)
        
    def test_apply_update_success(self):
        """测试应用模型更新成功"""
        update = {
            'param1': torch.ones(2, 2),
            'param2': torch.ones(3) * 0.5
        }
        
        config = {'learning_rate': 0.1}
        updated_model = self.strategy.apply_update(self.old_model, update, config)
        
        # 验证更新后模型
        expected_param1 = torch.ones(2, 2) + 0.1 * torch.ones(2, 2)  # 1 + 0.1 * 1
        expected_param2 = torch.ones(3) * 2 + 0.1 * torch.ones(3) * 0.5  # 2 + 0.1 * 0.5
        
        torch.testing.assert_close(updated_model['param1'], expected_param1)
        torch.testing.assert_close(updated_model['param2'], expected_param2)
        
    def test_apply_update_default_learning_rate(self):
        """测试使用默认学习率应用更新"""
        update = {
            'param1': torch.ones(2, 2),
            'param2': torch.ones(3) * 0.5
        }
        
        # 不提供配置，应该使用默认学习率1.0
        updated_model = self.strategy.apply_update(self.old_model, update)
        
        expected_param1 = torch.ones(2, 2) * 2  # 1 + 1.0 * 1
        expected_param2 = torch.ones(3) * 2.5  # 2 + 1.0 * 0.5
        
        torch.testing.assert_close(updated_model['param1'], expected_param1)
        torch.testing.assert_close(updated_model['param2'], expected_param2)
        
    def test_abstract_methods_enforcement(self):
        """测试抽象方法强制实现"""
        with self.assertRaises(TypeError):
            ModelUpdateStrategyInterface()


class TestPrivacyStrategyInterface(unittest.TestCase):
    
    def test_abstract_methods_enforcement(self):
        """测试隐私策略接口抽象方法强制实现"""
        with self.assertRaises(TypeError):
            PrivacyStrategyInterface()


class TestCommunicationStrategyInterface(unittest.TestCase):
    
    def test_abstract_methods_enforcement(self):
        """测试通信策略接口抽象方法强制实现"""
        with self.assertRaises(TypeError):
            CommunicationStrategyInterface()


class TestStrategyRegisterDecorator(unittest.TestCase):
    
    def test_strategy_register_decorator(self):
        """测试策略注册装饰器"""
        
        @strategy_register("aggregation", "test_aggregation")
        class TestAggregationStrategy:
            def aggregate(self, models, weights):
                return {}
                
        # 验证装饰器设置了属性
        self.assertEqual(TestAggregationStrategy._strategy_type, "aggregation")
        self.assertEqual(TestAggregationStrategy._strategy_name, "test_aggregation")


class TestStrategyErrorScenarios(unittest.TestCase):
    
    def test_strategy_error_instantiation(self):
        """测试策略错误实例化"""
        error = StrategyError("Test strategy error")
        
        self.assertIsInstance(error, Exception)
        self.assertEqual(str(error), "Test strategy error")
        
    def test_strategy_error_with_context(self):
        """测试带上下文的策略错误"""
        try:
            raise StrategyError("Strategy failed", error_code="STRAT001")
        except StrategyError as e:
            self.assertEqual(str(e), "Strategy failed")


if __name__ == '__main__':
    unittest.main()