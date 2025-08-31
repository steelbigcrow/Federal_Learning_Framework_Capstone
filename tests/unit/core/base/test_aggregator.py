"""
AbstractAggregator 抽象类全面单元测试
"""

import unittest
from unittest.mock import Mock, patch
import torch
import time

from src.core.base.aggregator import AbstractAggregator, AggregationMode
from src.core.base.component import ComponentStatus
from src.core.exceptions import AggregationError, AggregationConfigurationError


class ConcreteAggregator(AbstractAggregator):
    """用于测试的具体聚合器实现"""
    
    def aggregate(self, client_models, client_weights, metadata=None):
        """简单的FedAvg聚合实现"""
        if not client_models:
            raise AggregationError("No models provided")
            
        total_weight = sum(client_weights)
        aggregated_state = {}
        
        for key in client_models[0].keys():
            aggregated_param = torch.zeros_like(client_models[0][key])
            for model, weight in zip(client_models, client_weights):
                aggregated_param += model[key] * (weight / total_weight)
            aggregated_state[key] = aggregated_param
            
        return aggregated_state
        
    def compute_aggregation_weights(self, client_weights, metadata=None):
        """计算标准化权重"""
        total = sum(client_weights)
        if total == 0:
            raise AggregationError("Total weight is zero")
        return [w / total for w in client_weights]


class TestAbstractAggregator(unittest.TestCase):
    
    def setUp(self):
        """测试前设置"""
        self.config = {
            'aggregation': {
                'strategy': 'fedavg',
                'min_clients': 2
            }
        }
        
        # 创建测试模型状态
        self.client_models = [
            {
                'layer1.weight': torch.ones(3, 4) * (i + 1),
                'layer1.bias': torch.ones(3) * (i + 1)
            }
            for i in range(3)
        ]
        
        self.client_weights = [100, 200, 100]  # 总权重400
        
    def test_aggregator_initialization_default(self):
        """测试聚合器默认初始化"""
        aggregator = ConcreteAggregator()
        
        # 验证默认值
        self.assertEqual(aggregator.aggregation_mode, AggregationMode.WEIGHTED_AVERAGE)
        self.assertEqual(aggregator.aggregation_count, 0)
        self.assertIsNone(aggregator.last_aggregation_time)
        self.assertEqual(len(aggregator.get_aggregation_history()), 0)
        self.assertEqual(aggregator.status, ComponentStatus.READY)
        
    def test_aggregator_initialization_custom(self):
        """测试聚合器自定义初始化"""
        aggregator = ConcreteAggregator(
            aggregation_mode=AggregationMode.SIMPLE_AVERAGE,
            config=self.config
        )
        
        self.assertEqual(aggregator.aggregation_mode, AggregationMode.SIMPLE_AVERAGE)
        self.assertEqual(aggregator.get_config_value('aggregation.strategy'), 'fedavg')
        
    def test_aggregation_mode_management(self):
        """测试聚合模式管理"""
        aggregator = ConcreteAggregator()
        
        # 初始模式
        self.assertEqual(aggregator.aggregation_mode, AggregationMode.WEIGHTED_AVERAGE)
        
        # 更改模式
        aggregator.set_aggregation_mode(AggregationMode.MEDIAN)
        self.assertEqual(aggregator.aggregation_mode, AggregationMode.MEDIAN)
        
        # 运行时不能更改模式
        aggregator._set_status(ComponentStatus.RUNNING)
        with self.assertRaises(AggregationError):
            aggregator.set_aggregation_mode(AggregationMode.SIMPLE_AVERAGE)
            
    def test_model_validation_success(self):
        """测试模型验证成功"""
        aggregator = ConcreteAggregator()
        
        # 验证兼容的模型
        aggregator.validate_models(self.client_models)
        
        # 单个模型也应该通过（带警告）
        aggregator.validate_models(self.client_models[:1])
        
    def test_model_validation_failures(self):
        """测试模型验证失败情况"""
        aggregator = ConcreteAggregator()
        
        # 空模型列表
        with self.assertRaises(AggregationError):
            aggregator.validate_models([])
            
        # 不同的参数键
        incompatible_models = [
            self.client_models[0],
            {'different_key': torch.ones(2, 2)}
        ]
        with self.assertRaises(AggregationError):
            aggregator.validate_models(incompatible_models)
            
        # 不同的参数形状
        incompatible_shapes = [
            self.client_models[0],
            {
                'layer1.weight': torch.ones(2, 3),  # 不同形状
                'layer1.bias': torch.ones(3)
            }
        ]
        with self.assertRaises(AggregationError):
            aggregator.validate_models(incompatible_shapes)
            
    def test_weight_validation_success(self):
        """测试权重验证成功"""
        aggregator = ConcreteAggregator()
        
        # 有效权重
        aggregator.validate_weights(self.client_models, self.client_weights)
        
        # 零权重也应该通过单个权重验证（但总和检查会失败）
        zero_weights = [0, 100, 50]
        aggregator.validate_weights(self.client_models, zero_weights)
        
    def test_weight_validation_failures(self):
        """测试权重验证失败情况"""
        aggregator = ConcreteAggregator()
        
        # 权重数量不匹配
        wrong_count_weights = [100, 200]  # 只有2个权重，但有3个模型
        with self.assertRaises(AggregationError):
            aggregator.validate_weights(self.client_models, wrong_count_weights)
            
        # 负权重
        negative_weights = [100, -50, 100]
        with self.assertRaises(AggregationError):
            aggregator.validate_weights(self.client_models, negative_weights)
            
        # 非数值权重
        invalid_weights = [100, "invalid", 100]
        with self.assertRaises(AggregationError):
            aggregator.validate_weights(self.client_models, invalid_weights)
            
        # 全零权重
        zero_weights = [0, 0, 0]
        with self.assertRaises(AggregationError):
            aggregator.validate_weights(self.client_models, zero_weights)
            
    def test_parameter_filtering(self):
        """测试参数过滤"""
        aggregator = ConcreteAggregator()
        
        model_state = self.client_models[0]
        
        # 无过滤器 - 返回所有参数
        filtered = aggregator.filter_parameters(model_state, None)
        self.assertEqual(set(filtered.keys()), set(model_state.keys()))
        
        # 过滤特定参数
        filter_set = {'layer1.weight'}
        filtered = aggregator.filter_parameters(model_state, filter_set)
        self.assertEqual(set(filtered.keys()), filter_set)
        
        # 过滤不存在的参数
        filter_set = {'nonexistent_param'}
        filtered = aggregator.filter_parameters(model_state, filter_set)
        self.assertEqual(len(filtered), 0)
        
    def test_basic_aggregation(self):
        """测试基本聚合功能"""
        aggregator = ConcreteAggregator()
        
        # 执行聚合
        result = aggregator.aggregate(self.client_models, self.client_weights)
        
        # 验证结果结构
        self.assertIn('layer1.weight', result)
        self.assertIn('layer1.bias', result)
        
        # 验证加权平均计算
        expected_weight = (torch.ones(3, 4) * 1 * 100/400 + 
                          torch.ones(3, 4) * 2 * 200/400 + 
                          torch.ones(3, 4) * 3 * 100/400)
        torch.testing.assert_close(result['layer1.weight'], expected_weight)
        
    def test_aggregation_with_validation(self):
        """测试带验证的聚合"""
        aggregator = ConcreteAggregator()
        
        # 正常聚合
        result = aggregator.aggregate_with_validation(
            self.client_models, self.client_weights
        )
        
        # 验证聚合历史更新
        self.assertEqual(aggregator.aggregation_count, 1)
        self.assertIsNotNone(aggregator.last_aggregation_time)
        
        history = aggregator.get_aggregation_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]['round'], 1)
        self.assertEqual(history[0]['num_clients'], 3)
        
        # 验证状态管理
        self.assertEqual(aggregator.status, ComponentStatus.READY)
        
    def test_aggregation_with_parameter_filter(self):
        """测试带参数过滤的聚合"""
        aggregator = ConcreteAggregator()
        
        # 只聚合权重参数
        parameter_filter = {'layer1.weight'}
        result = aggregator.aggregate_with_validation(
            self.client_models, self.client_weights,
            parameter_filter=parameter_filter
        )
        
        # 验证只有过滤的参数被聚合
        self.assertEqual(set(result.keys()), parameter_filter)
        
    def test_aggregation_error_handling(self):
        """测试聚合错误处理"""
        aggregator = ConcreteAggregator()
        
        # 模拟聚合过程中的错误
        with patch.object(aggregator, 'aggregate', side_effect=Exception("Aggregation failed")):
            with self.assertRaises(AggregationError):
                aggregator.aggregate_with_validation(self.client_models, self.client_weights)
                
            # 验证状态变为错误
            self.assertEqual(aggregator.status, ComponentStatus.ERROR)
            
    def test_aggregation_statistics(self):
        """测试聚合统计信息"""
        aggregator = ConcreteAggregator()
        
        # 初始时无统计
        stats = aggregator.get_aggregation_stats()
        self.assertEqual(stats['total_aggregations'], 0)
        
        # 执行多次聚合
        for _ in range(3):
            aggregator.aggregate_with_validation(self.client_models, self.client_weights)
            
        # 验证统计
        stats = aggregator.get_aggregation_stats()
        self.assertEqual(stats['total_aggregations'], 3)
        self.assertIn('avg_aggregation_time', stats)
        self.assertIn('min_aggregation_time', stats)
        self.assertIn('max_aggregation_time', stats)
        self.assertIn('total_aggregation_time', stats)
        
    def test_weight_computation(self):
        """测试权重计算"""
        aggregator = ConcreteAggregator()
        
        # 标准化权重
        normalized = aggregator.compute_aggregation_weights(self.client_weights)
        
        # 验证权重和为1
        self.assertAlmostEqual(sum(normalized), 1.0, places=6)
        
        # 验证权重比例
        expected = [100/400, 200/400, 100/400]
        for actual, exp in zip(normalized, expected):
            self.assertAlmostEqual(actual, exp, places=6)
            
    def test_weight_computation_zero_total(self):
        """测试零总权重的权重计算"""
        aggregator = ConcreteAggregator()
        
        with self.assertRaises(AggregationError):
            aggregator.compute_aggregation_weights([0, 0, 0])
            
    def test_aggregator_reset(self):
        """测试聚合器重置"""
        aggregator = ConcreteAggregator()
        
        # 执行一些聚合
        aggregator.aggregate_with_validation(self.client_models, self.client_weights)
        aggregator.aggregate_with_validation(self.client_models, self.client_weights)
        
        # 验证状态不为空
        self.assertGreater(aggregator.aggregation_count, 0)
        self.assertGreater(len(aggregator.get_aggregation_history()), 0)
        
        # 重置
        aggregator.reset()
        
        # 验证重置后状态
        self.assertEqual(aggregator.aggregation_count, 0)
        self.assertIsNone(aggregator.last_aggregation_time)
        self.assertEqual(len(aggregator.get_aggregation_history()), 0)
        self.assertEqual(aggregator.status, ComponentStatus.READY)
        
    def test_multiple_aggregation_rounds(self):
        """测试多轮聚合"""
        aggregator = ConcreteAggregator()
        
        # 执行多轮聚合，每轮使用不同的权重
        weights_list = [
            [100, 100, 100],  # 均等权重
            [300, 100, 100],  # 第一个客户端权重更大
            [100, 100, 300]   # 第三个客户端权重更大
        ]
        
        results = []
        for weights in weights_list:
            result = aggregator.aggregate_with_validation(self.client_models, weights)
            results.append(result)
            
        # 验证聚合历史
        self.assertEqual(aggregator.aggregation_count, 3)
        history = aggregator.get_aggregation_history()
        self.assertEqual(len(history), 3)
        
        # 验证不同权重产生不同结果
        self.assertFalse(torch.equal(results[0]['layer1.weight'], results[1]['layer1.weight']))
        self.assertFalse(torch.equal(results[1]['layer1.weight'], results[2]['layer1.weight']))
        
    def test_aggregation_with_metadata(self):
        """测试带元数据的聚合"""
        aggregator = ConcreteAggregator()
        
        metadata = [
            {'client_id': 1, 'accuracy': 0.8},
            {'client_id': 2, 'accuracy': 0.9},
            {'client_id': 3, 'accuracy': 0.7}
        ]
        
        # 执行带元数据的聚合
        result = aggregator.aggregate_with_validation(
            self.client_models, self.client_weights, metadata=metadata
        )
        
        # 验证结果仍然正确
        self.assertIn('layer1.weight', result)
        
    def test_string_representation(self):
        """测试字符串表示"""
        aggregator = ConcreteAggregator(
            aggregation_mode=AggregationMode.MEDIAN
        )
        
        str_repr = str(aggregator)
        self.assertIn("Aggregator", str_repr)
        self.assertIn("mode=median", str_repr)
        self.assertIn("count=0", str_repr)
        
        # 执行聚合后
        aggregator.aggregate_with_validation(self.client_models, self.client_weights)
        
        str_repr = str(aggregator)
        self.assertIn("count=1", str_repr)


class TestAggregationMode(unittest.TestCase):
    """测试聚合模式枚举"""
    
    def test_aggregation_mode_values(self):
        """测试聚合模式枚举值"""
        expected_modes = [
            "weighted_average", "simple_average", "median", 
            "trimmed_mean", "fedprox", "custom"
        ]
        
        actual_modes = [mode.value for mode in AggregationMode]
        
        for expected in expected_modes:
            self.assertIn(expected, actual_modes)


class TestAbstractAggregatorAbstractMethods(unittest.TestCase):
    """测试抽象方法强制实现"""
    
    def test_abstract_methods_enforcement(self):
        """测试抽象方法必须被实现"""
        
        # 尝试直接实例化抽象类应该失败
        with self.assertRaises(TypeError):
            AbstractAggregator()


if __name__ == '__main__':
    unittest.main()