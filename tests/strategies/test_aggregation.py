"""
聚合策略单元测试
"""

import pytest
import torch
from typing import Dict, List, Any

from src.strategies.aggregation.fedavg import FedAvgStrategy
from src.strategies.aggregation.lora_fedavg import LoRAFedAvgStrategy
from src.strategies.aggregation.adalora_fedavg import AdaLoRAFedAvgStrategy
from src.core.interfaces.strategy import StrategyType
from src.core.exceptions.exceptions import StrategyError


class TestFedAvgStrategy:
    """测试标准联邦平均策略"""
    
    def setup_method(self):
        """测试前设置"""
        self.strategy = FedAvgStrategy()
    
    def test_get_name(self):
        """测试获取策略名称"""
        assert self.strategy.get_name() == "fedavg"
    
    def test_get_description(self):
        """测试获取策略描述"""
        description = self.strategy.get_description()
        assert "标准联邦平均" in description
    
    def test_validate_context_valid(self):
        """测试验证有效上下文"""
        context = {
            'client_models': [{'param1': torch.tensor([1.0])}],
            'client_weights': [1.0]
        }
        # 应该不抛出异常
        self.strategy.validate_context(context)
    
    def test_validate_context_missing_key(self):
        """测试验证缺少键的上下文"""
        context = {'client_models': [{'param1': torch.tensor([1.0])}]}
        with pytest.raises(StrategyError, match="缺少必需的上下文键"):
            self.strategy.validate_context(context)
    
    def test_validate_context_empty_models(self):
        """测试验证空模型列表"""
        context = {
            'client_models': [],
            'client_weights': [1.0]
        }
        with pytest.raises(StrategyError, match="client_models 必须是非空列表"):
            self.strategy.validate_context(context)
    
    def test_aggregate_models_basic(self):
        """测试基本模型聚合"""
        # 创建模拟模型参数
        model1 = {'layer1.weight': torch.tensor([[1.0, 2.0], [3.0, 4.0]])}
        model2 = {'layer1.weight': torch.tensor([[5.0, 6.0], [7.0, 8.0]])}
        
        weights = [1.0, 1.0]
        
        result = self.strategy.aggregate_models([model1, model2], weights)
        
        expected = torch.tensor([[3.0, 4.0], [5.0, 6.0]])  # 平均值
        assert torch.allclose(result['layer1.weight'], expected)
    
    def test_aggregate_models_unequal_weights(self):
        """测试不等权重聚合"""
        model1 = {'layer1.weight': torch.tensor([[1.0, 1.0]])}
        model2 = {'layer1.weight': torch.tensor([[3.0, 3.0]])}
        
        weights = [2.0, 1.0]  # model1权重是model2的两倍
        
        result = self.strategy.aggregate_models([model1, model2], weights)
        
        expected = torch.tensor([[1.6667, 1.6667]])  # (2*1 + 1*3) / 3
        assert torch.allclose(result['layer1.weight'], expected, atol=1e-4)
    
    def test_aggregate_models_zero_total_weight(self):
        """测试总权重为0的情况"""
        model1 = {'layer1.weight': torch.tensor([[1.0, 1.0]])}
        model2 = {'layer1.weight': torch.tensor([[2.0, 2.0]])}
        
        weights = [0.0, 0.0]
        
        with pytest.raises(StrategyError, match="客户端权重总和为0"):
            self.strategy.aggregate_models([model1, model2], weights)
    
    def test_aggregate_models_missing_param(self):
        """测试模型缺少参数的情况"""
        model1 = {'layer1.weight': torch.tensor([[1.0]])}
        model2 = {'layer2.weight': torch.tensor([[2.0]])}  # 不同的参数名
        
        weights = [1.0, 1.0]
        
        with pytest.raises(StrategyError, match="客户端模型缺少参数"):
            self.strategy.aggregate_models([model1, model2], weights)
    
    def test_compute_weights_with_samples(self):
        """测试基于样本数量的权重计算"""
        client_metrics = [
            {'num_samples': 100},
            {'num_samples': 200},
            {'num_samples': 300}
        ]
        
        weights = self.strategy.compute_weights(client_metrics)
        
        expected = [100/600, 200/600, 300/600]  # [0.1667, 0.3333, 0.5]
        assert len(weights) == 3
        assert sum(weights) == pytest.approx(1.0)
        assert weights == pytest.approx(expected, rel=1e-4)
    
    def test_compute_weights_equal(self):
        """测试等权重计算"""
        client_metrics = [{'num_samples': 100}, {'num_samples': 100}]
        
        weights = self.strategy.compute_weights(client_metrics)
        
        assert weights == [0.5, 0.5]
    
    def test_compute_weights_no_samples(self):
        """测试没有样本数量的情况"""
        client_metrics = [{'loss': 0.5}, {'loss': 0.3}]
        
        weights = self.strategy.compute_weights(client_metrics)
        
        assert weights == [0.5, 0.5]  # 等权重
    
    def test_compute_weights_empty(self):
        """测试空指标列表"""
        weights = self.strategy.compute_weights([])
        assert weights == []
    
    def test_supports_parameter_filtering(self):
        """测试参数过滤支持"""
        assert not self.strategy.supports_parameter_filtering()
    
    def test_get_required_metrics(self):
        """测试获取必需指标"""
        metrics = self.strategy.get_required_metrics()
        assert "num_samples" in metrics
    
    def test_get_config_schema(self):
        """测试获取配置模式"""
        schema = self.strategy.get_config_schema()
        assert isinstance(schema, dict)
        assert "type" in schema
        assert "properties" in schema


class TestLoRAFedAvgStrategy:
    """测试LoRA联邦平均策略"""
    
    def setup_method(self):
        """测试前设置"""
        self.strategy = LoRAFedAvgStrategy()
    
    def test_get_name(self):
        """测试获取策略名称"""
        assert self.strategy.get_name() == "lora_fedavg"
    
    def test_get_description(self):
        """测试获取策略描述"""
        description = self.strategy.get_description()
        assert "LoRA联邦平均" in description
    
    def test_aggregate_models_with_trainable_keys(self):
        """测试使用可训练键的聚合"""
        model1 = {
            'lora_A.weight': torch.tensor([[1.0]]),
            'lora_B.weight': torch.tensor([[2.0]]),
            'base.weight': torch.tensor([[10.0]])  # 基础参数
        }
        model2 = {
            'lora_A.weight': torch.tensor([[3.0]]),
            'lora_B.weight': torch.tensor([[4.0]]),
            'base.weight': torch.tensor([[20.0]])  # 基础参数
        }
        
        weights = [1.0, 1.0]
        trainable_keys = {'lora_A.weight', 'lora_B.weight'}
        
        result = self.strategy.aggregate_models([model1, model2], weights, trainable_keys=trainable_keys)
        
        # LoRA参数应该被聚合
        assert torch.allclose(result['lora_A.weight'], torch.tensor([[2.0]]))
        assert torch.allclose(result['lora_B.weight'], torch.tensor([[3.0]]))
        
        # 基础参数应该直接使用第一个客户端的值
        assert torch.allclose(result['base.weight'], torch.tensor([[10.0]]))
    
    def test_aggregate_models_without_trainable_keys(self):
        """测试不指定可训练键的聚合"""
        model1 = {'lora_A.weight': torch.tensor([[1.0]]), 'base.weight': torch.tensor([[10.0]])}
        model2 = {'lora_A.weight': torch.tensor([[3.0]]), 'base.weight': torch.tensor([[20.0]])}
        
        weights = [1.0, 1.0]
        
        result = self.strategy.aggregate_models([model1, model2], weights)
        
        # 所有参数都应该被聚合（向后兼容）
        assert torch.allclose(result['lora_A.weight'], torch.tensor([[2.0]]))
        assert torch.allclose(result['base.weight'], torch.tensor([[15.0]]))
    
    def test_supports_parameter_filtering(self):
        """测试参数过滤支持"""
        assert self.strategy.supports_parameter_filtering()
    
    def test_get_trainable_keys(self):
        """测试获取可训练键"""
        # 创建一个模拟模型
        model = torch.nn.Linear(10, 5)
        model.weight.requires_grad = True
        model.bias.requires_grad = False
        
        trainable_keys = self.strategy.get_trainable_keys(model)
        
        assert 'weight' in trainable_keys
        assert 'bias' not in trainable_keys
    
    def test_validate_context_missing_client_models(self):
        """测试验证缺少client_models的上下文"""
        context = {'client_weights': [1.0]}
        with pytest.raises(StrategyError, match="缺少必需的上下文键: client_models"):
            self.strategy.validate_context(context)
    
    def test_validate_context_missing_client_weights(self):
        """测试验证缺少client_weights的上下文"""
        context = {'client_models': [{'param': torch.tensor([1.0])}]}
        with pytest.raises(StrategyError, match="缺少必需的上下文键: client_weights"):
            self.strategy.validate_context(context)
    
    def test_validate_context_empty_client_models(self):
        """测试验证空client_models列表"""
        context = {'client_models': [], 'client_weights': [1.0]}
        with pytest.raises(StrategyError, match="client_models 必须是非空列表"):
            self.strategy.validate_context(context)
    
    def test_validate_context_mismatched_lengths(self):
        """测试验证client_models和client_weights长度不匹配"""
        context = {
            'client_models': [{'param': torch.tensor([1.0])}, {'param': torch.tensor([2.0])}],
            'client_weights': [1.0]
        }
        with pytest.raises(StrategyError, match="client_weights 必须是与 client_models 等长的列表"):
            self.strategy.validate_context(context)
    
    def test_aggregate_models_empty_list(self):
        """测试聚合空模型列表"""
        with pytest.raises(StrategyError, match="客户端模型列表为空"):
            self.strategy.aggregate_models([], [1.0])
    
    def test_aggregate_models_mismatched_lengths(self):
        """测试聚合时模型和权重长度不匹配"""
        model1 = {'param': torch.tensor([1.0])}
        model2 = {'param': torch.tensor([2.0])}
        
        with pytest.raises(StrategyError, match="客户端模型数量与权重数量不匹配"):
            self.strategy.aggregate_models([model1, model2], [1.0])
    
    def test_execute_with_context(self):
        """测试使用完整上下文执行策略"""
        context = {
            'client_models': [{'lora_weight': torch.tensor([1.0])}, {'lora_weight': torch.tensor([3.0])}],
            'client_weights': [1.0, 1.0],
            'global_model': {'lora_weight': torch.tensor([0.0])},
            'trainable_keys': {'lora_weight'}
        }
        
        result = self.strategy.execute(context)
        
        # 应该返回聚合结果
        assert 'lora_weight' in result
        assert torch.allclose(result['lora_weight'], torch.tensor([2.0]))
    
    def test_execute_minimal_context(self):
        """测试使用最小上下文执行策略"""
        context = {
            'client_models': [{'lora_weight': torch.tensor([1.0])}, {'lora_weight': torch.tensor([3.0])}],
            'client_weights': [1.0, 1.0]
        }
        
        result = self.strategy.execute(context)
        
        # 应该返回聚合结果
        assert 'lora_weight' in result
        assert torch.allclose(result['lora_weight'], torch.tensor([2.0]))


class TestAdaLoRAFedAvgStrategy:
    """测试AdaLoRA联邦平均策略"""
    
    def setup_method(self):
        """测试前设置"""
        self.strategy = AdaLoRAFedAvgStrategy()
    
    def test_get_name(self):
        """测试获取策略名称"""
        assert self.strategy.get_name() == "adalora_fedavg"
    
    def test_get_description(self):
        """测试获取策略描述"""
        description = self.strategy.get_description()
        assert "AdaLoRA联邦平均" in description
    
    def test_aggregate_models_partial_params(self):
        """测试部分参数聚合"""
        model1 = {
            'adalora_A.weight': torch.tensor([[1.0]]),
            'adalora_B.weight': torch.tensor([[2.0]]),
            'classifier.weight': torch.tensor([[10.0]])
        }
        model2 = {
            'adalora_A.weight': torch.tensor([[3.0]]),
            'adalora_B.weight': torch.tensor([[4.0]]),
            # model2没有classifier.weight
        }
        
        weights = [1.0, 1.0]
        trainable_keys = {'adalora_A.weight', 'adalora_B.weight', 'classifier.weight'}
        
        result = self.strategy.aggregate_models([model1, model2], weights, trainable_keys=trainable_keys)
        
        # 存在于两个模型中的参数应该被聚合
        assert torch.allclose(result['adalora_A.weight'], torch.tensor([[2.0]]))
        assert torch.allclose(result['adalora_B.weight'], torch.tensor([[3.0]]))
        
        # 只存在于一个模型中的参数应该被跳过
        assert 'classifier.weight' not in result
    
    def test_aggregate_with_svd(self):
        """测试SVD聚合方法"""
        model1 = {'param': torch.tensor([[1.0, 2.0], [3.0, 4.0]])}
        model2 = {'param': torch.tensor([[5.0, 6.0], [7.0, 8.0]])}
        
        weights = [1.0, 1.0]
        
        result = self.strategy.aggregate_with_svd([model1, model2], weights, 'param')
        
        expected = torch.tensor([[3.0, 4.0], [5.0, 6.0]])  # 平均值
        assert torch.allclose(result, expected)
    
    def test_aggregate_with_svd_zero_weights(self):
        """测试SVD聚合权重为0的情况"""
        model1 = {'param': torch.tensor([[1.0]])}
        model2 = {'param': torch.tensor([[2.0]])}
        
        weights = [0.0, 0.0]
        
        result = self.strategy.aggregate_with_svd([model1, model2], weights, 'param')
        
        # 权重为0时应该使用简单平均
        expected = torch.tensor([[1.5]])
        assert torch.allclose(result, expected)
    
    def test_validate_context_missing_client_models(self):
        """测试验证缺少client_models的上下文"""
        context = {'client_weights': [1.0]}
        with pytest.raises(StrategyError, match="缺少必需的上下文键: client_models"):
            self.strategy.validate_context(context)
    
    def test_validate_context_missing_client_weights(self):
        """测试验证缺少client_weights的上下文"""
        context = {'client_models': [{'param': torch.tensor([1.0])}]}
        with pytest.raises(StrategyError, match="缺少必需的上下文键: client_weights"):
            self.strategy.validate_context(context)
    
    def test_validate_context_empty_client_models(self):
        """测试验证空client_models列表"""
        context = {'client_models': [], 'client_weights': [1.0]}
        with pytest.raises(StrategyError, match="client_models 必须是非空列表"):
            self.strategy.validate_context(context)
    
    def test_validate_context_mismatched_lengths(self):
        """测试验证client_models和client_weights长度不匹配"""
        context = {
            'client_models': [{'param': torch.tensor([1.0])}, {'param': torch.tensor([2.0])}],
            'client_weights': [1.0]
        }
        with pytest.raises(StrategyError, match="client_weights 必须是与 client_models 等长的列表"):
            self.strategy.validate_context(context)
    
    def test_aggregate_models_empty_list(self):
        """测试聚合空模型列表"""
        with pytest.raises(StrategyError, match="客户端模型列表为空"):
            self.strategy.aggregate_models([], [1.0])
    
    def test_aggregate_models_mismatched_lengths(self):
        """测试聚合时模型和权重长度不匹配"""
        model1 = {'param': torch.tensor([1.0])}
        model2 = {'param': torch.tensor([2.0])}
        
        with pytest.raises(StrategyError, match="客户端模型数量与权重数量不匹配"):
            self.strategy.aggregate_models([model1, model2], [1.0])
    
    def test_execute_with_context(self):
        """测试使用完整上下文执行策略"""
        context = {
            'client_models': [{'param': torch.tensor([1.0])}, {'param': torch.tensor([3.0])}],
            'client_weights': [1.0, 1.0],
            'global_model': {'param': torch.tensor([0.0])},
            'trainable_keys': {'param'}
        }
        
        result = self.strategy.execute(context)
        
        # 应该返回聚合结果
        assert 'param' in result
        assert torch.allclose(result['param'], torch.tensor([2.0]))
    
    def test_execute_minimal_context(self):
        """测试使用最小上下文执行策略"""
        context = {
            'client_models': [{'param': torch.tensor([1.0])}, {'param': torch.tensor([3.0])}],
            'client_weights': [1.0, 1.0]
        }
        
        result = self.strategy.execute(context)
        
        # 应该返回聚合结果
        assert 'param' in result
        assert torch.allclose(result['param'], torch.tensor([2.0]))


class TestStrategyIntegration:
    """测试策略集成"""
    
    def test_all_strategies_registered(self):
        """测试所有策略都已注册"""
        from src.core.interfaces.strategy import StrategyRegistry
        
        # 检查聚合策略
        agg_strategies = StrategyRegistry.list_strategies(StrategyType.AGGREGATION)
        assert "fedavg" in agg_strategies
        assert "lora_fedavg" in agg_strategies
        assert "adalora_fedavg" in agg_strategies
        
        # 检查训练策略
        train_strategies = StrategyRegistry.list_strategies(StrategyType.TRAINING)
        assert "standard" in train_strategies
        assert "lora" in train_strategies
        assert "adalora" in train_strategies
    
    def test_strategy_factory(self):
        """测试策略工厂"""
        from src.strategies.strategy_factory import strategy_factory
        from src.strategies.training.standard import StandardTrainingStrategy
        
        # 测试创建聚合策略
        fedavg = strategy_factory.create_aggregation_strategy("fedavg")
        assert isinstance(fedavg, FedAvgStrategy)
        
        # 测试创建训练策略
        standard = strategy_factory.create_training_strategy("standard")
        assert isinstance(standard, StandardTrainingStrategy)
        
        # 测试获取可用策略
        agg_strategies = strategy_factory.get_available_strategies(StrategyType.AGGREGATION)
        assert len(agg_strategies) >= 3  # 至少有3个聚合策略
    
    def test_strategy_config_validation(self):
        """测试策略配置验证"""
        from src.strategies.strategy_factory import strategy_factory
        
        # 有效配置
        valid_config = {
            'client_models': [{'param': torch.tensor([1.0])}],
            'client_weights': [1.0]
        }
        
        assert strategy_factory.validate_strategy_config(
            StrategyType.AGGREGATION, "fedavg", valid_config
        )
        
        # 无效配置
        invalid_config = {'client_models': []}
        
        assert not strategy_factory.validate_strategy_config(
            StrategyType.AGGREGATION, "fedavg", invalid_config
        )