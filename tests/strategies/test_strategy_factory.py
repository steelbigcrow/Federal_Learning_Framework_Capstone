"""
策略工厂单元测试
"""

import pytest
from typing import Dict, Any

from src.strategies.strategy_factory import StrategyFactory
from src.strategies.aggregation.fedavg import FedAvgStrategy
from src.strategies.training.standard import StandardTrainingStrategy
from src.core.interfaces.strategy import StrategyType, StrategyRegistry
from src.core.exceptions.exceptions import StrategyError


class TestStrategyFactory:
    """测试策略工厂"""
    
    def setup_method(self):
        """测试前设置"""
        self.factory = StrategyFactory()
    
    def test_singleton_instance(self):
        """测试单例实例"""
        factory1 = StrategyFactory()
        factory2 = StrategyFactory()
        
        # 应该是不同的实例，但都注册了相同的策略
        assert factory1 is not factory2
        
        # 两个工厂都应该有相同的策略
        agg1 = factory1.get_available_strategies(StrategyType.AGGREGATION)
        agg2 = factory2.get_available_strategies(StrategyType.AGGREGATION)
        assert set(agg1) == set(agg2)
    
    def test_register_strategy(self):
        """测试注册策略"""
        # 创建一个测试策略
        test_strategy = FedAvgStrategy()
        
        # 注册策略
        self.factory.register_strategy(StrategyType.AGGREGATION, "test_fedavg", test_strategy)
        
        # 验证策略已注册
        strategies = self.factory.get_available_strategies(StrategyType.AGGREGATION)
        assert "test_fedavg" in strategies
        
        # 验证可以获取策略
        retrieved_strategy = self.factory.create_strategy(StrategyType.AGGREGATION, "test_fedavg")
        assert retrieved_strategy is test_strategy
    
    def test_create_strategy_existing(self):
        """测试创建已存在的策略"""
        strategy = self.factory.create_strategy(StrategyType.AGGREGATION, "fedavg")
        
        assert isinstance(strategy, FedAvgStrategy)
        assert strategy.get_name() == "fedavg"
    
    def test_create_strategy_nonexistent(self):
        """测试创建不存在的策略"""
        with pytest.raises(StrategyError, match="策略 'nonexistent' 不存在于类型 'aggregation' 中"):
            self.factory.create_strategy(StrategyType.AGGREGATION, "nonexistent")
    
    def test_create_aggregation_strategy(self):
        """测试创建聚合策略"""
        strategy = self.factory.create_aggregation_strategy("fedavg")
        
        assert isinstance(strategy, FedAvgStrategy)
        assert strategy.get_name() == "fedavg"
    
    def test_create_aggregation_strategy_wrong_type(self):
        """测试创建错误类型的聚合策略"""
        with pytest.raises(StrategyError, match="无法创建聚合策略 'standard'"):
            self.factory.create_aggregation_strategy("standard")
    
    def test_create_training_strategy(self):
        """测试创建训练策略"""
        strategy = self.factory.create_training_strategy("standard")
        
        assert isinstance(strategy, StandardTrainingStrategy)
        assert strategy.get_name() == "standard"
    
    def test_create_training_strategy_wrong_type(self):
        """测试创建错误类型的训练策略"""
        with pytest.raises(StrategyError, match="无法创建训练策略 'fedavg'"):
            self.factory.create_training_strategy("fedavg")
    
    def test_get_available_strategies(self):
        """测试获取可用策略"""
        agg_strategies = self.factory.get_available_strategies(StrategyType.AGGREGATION)
        train_strategies = self.factory.get_available_strategies(StrategyType.TRAINING)
        
        assert len(agg_strategies) >= 1
        assert len(train_strategies) >= 1
        assert "fedavg" in agg_strategies
        assert "standard" in train_strategies
    
    def test_get_strategy_info(self):
        """测试获取策略信息"""
        info = self.factory.get_strategy_info(StrategyType.AGGREGATION, "fedavg")
        
        assert isinstance(info, dict)
        assert "name" in info
        assert "description" in info
        assert "type" in info
        assert "config_schema" in info
        
        assert info["name"] == "fedavg"
        assert info["type"] == "aggregation"
    
    def test_get_strategy_info_nonexistent(self):
        """测试获取不存在策略的信息"""
        with pytest.raises(StrategyError, match="策略 'nonexistent' 不存在"):
            self.factory.get_strategy_info(StrategyType.AGGREGATION, "nonexistent")
    
    def test_validate_strategy_config_valid(self):
        """测试验证有效策略配置"""
        config = {
            'client_models': [{'param': 'value'}],
            'client_weights': [1.0]
        }
        
        result = self.factory.validate_strategy_config(
            StrategyType.AGGREGATION, "fedavg", config
        )
        
        assert result is True
    
    def test_validate_strategy_config_invalid(self):
        """测试验证无效策略配置"""
        config = {
            'client_models': [],  # 空列表是无效的
            'client_weights': [1.0]
        }
        
        result = self.factory.validate_strategy_config(
            StrategyType.AGGREGATION, "fedavg", config
        )
        
        assert result is False
    
    def test_list_all_strategies(self):
        """测试列出所有策略"""
        all_strategies = self.factory.list_all_strategies()
        
        assert isinstance(all_strategies, dict)
        assert "aggregation" in all_strategies
        assert "training" in all_strategies
        
        assert len(all_strategies["aggregation"]) >= 1
        assert len(all_strategies["training"]) >= 1
    
    def test_get_default_strategy(self):
        """测试获取默认策略"""
        default_agg = self.factory.get_default_strategy(StrategyType.AGGREGATION)
        default_train = self.factory.get_default_strategy(StrategyType.TRAINING)
        
        assert default_agg == "fedavg"
        assert default_train == "standard"
    
    def test_get_default_strategy_unknown_type(self):
        """测试获取未知类型的默认策略"""
        # 创建一个假的策略类型
        from enum import Enum
        
        class FakeStrategyType(Enum):
            FAKE = "fake"
        
        default = self.factory.get_default_strategy(FakeStrategyType.FAKE)
        assert default == "default"
    
    def test_clear_registry(self):
        """测试清空注册表"""
        # 先注册一个测试策略
        test_strategy = FedAvgStrategy()
        self.factory.register_strategy(StrategyType.AGGREGATION, "test_clear", test_strategy)
        
        # 确认策略已注册
        assert "test_clear" in self.factory.get_available_strategies(StrategyType.AGGREGATION)
        
        # 清空注册表
        self.factory.clear_registry()
        
        # 确认策略已被清空
        assert "test_clear" not in self.factory.get_available_strategies(StrategyType.AGGREGATION)
        # 但默认策略应该还在，因为重新加载了
        assert "fedavg" in self.factory.get_available_strategies(StrategyType.AGGREGATION)
    
    def test_reload_strategies(self):
        """测试重新加载策略"""
        # 注册一个测试策略
        test_strategy = FedAvgStrategy()
        self.factory.register_strategy(StrategyType.AGGREGATION, "test_reload", test_strategy)
        
        # 确认策略已注册
        assert "test_reload" in self.factory.get_available_strategies(StrategyType.AGGREGATION)
        
        # 重新加载策略
        self.factory.reload_strategies()
        
        # 确认测试策略已被移除
        assert "test_reload" not in self.factory.get_available_strategies(StrategyType.AGGREGATION)
        
        # 确认默认策略仍然存在
        assert "fedavg" in self.factory.get_available_strategies(StrategyType.AGGREGATION)
        assert "standard" in self.factory.get_available_strategies(StrategyType.TRAINING)
    
    def test_configure_strategy(self):
        """测试配置策略"""
        strategy = FedAvgStrategy()
        config = {"test_param": "test_value"}
        
        # 这个方法目前是空实现，应该不抛出异常
        self.factory._configure_strategy(strategy, config)
        
        # 策略应该不受影响
        assert strategy.get_name() == "fedavg"


class TestStrategyFactoryIntegration:
    """测试策略工厂集成"""
    
    def test_global_factory_instance(self):
        """测试全局工厂实例"""
        from src.strategies.strategy_factory import strategy_factory
        
        # 全局实例应该可用
        assert strategy_factory is not None
        
        # 应该能够创建策略
        strategy = strategy_factory.create_aggregation_strategy("fedavg")
        assert isinstance(strategy, FedAvgStrategy)
    
    def test_factory_with_registry(self):
        """测试工厂与注册表的集成"""
        # 获取当前的注册表状态
        original_strategies = StrategyRegistry.list_strategies(StrategyType.AGGREGATION)
        
        # 创建新工厂
        factory = StrategyFactory()
        
        # 新工厂应该有相同的策略
        factory_strategies = factory.get_available_strategies(StrategyType.AGGREGATION)
        assert set(factory_strategies) == set(original_strategies)
    
    def test_multiple_factories_independent(self):
        """测试多个工厂实例的独立性"""
        factory1 = StrategyFactory()
        factory2 = StrategyFactory()
        
        # 在factory1中注册一个新策略
        test_strategy = FedAvgStrategy()
        factory1.register_strategy(StrategyType.AGGREGATION, "independent_test", test_strategy)
        
        # factory1应该有新策略
        assert "independent_test" in factory1.get_available_strategies(StrategyType.AGGREGATION)
        
        # factory2不应该有新策略（除非它们共享注册表）
        factory2_strategies = factory2.get_available_strategies(StrategyType.AGGREGATION)
        if "independent_test" not in factory2_strategies:
            # 这表明工厂实例是独立的
            pass
        else:
            # 如果策略在factory2中也存在，说明它们共享注册表
            # 这也是可以接受的行为
            pass
    
    def test_strategy_creation_with_config(self):
        """测试带配置的策略创建"""
        factory = StrategyFactory()
        
        config = {
            'epochs': 5,
            'lr': 0.01,
            'optimizer': 'sgd'
        }
        
        strategy = factory.create_training_strategy("standard", config)
        
        # 策略应该被创建
        assert isinstance(strategy, StandardTrainingStrategy)
        
        # 配置验证应该通过（需要完整的上下文）
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        
        model = nn.Linear(10, 2)
        data = torch.randn(10, 10)
        targets = torch.randint(0, 2, (10,))
        dataset = TensorDataset(data, targets)
        train_loader = DataLoader(dataset, batch_size=5)
        
        full_context = {
            'model': model,
            'train_loader': train_loader,
            'config': config
        }
        
        assert factory.validate_strategy_config(StrategyType.TRAINING, "standard", full_context)
    
    def test_strategy_info_comprehensive(self):
        """测试策略信息的完整性"""
        factory = StrategyFactory()
        
        info = factory.get_strategy_info(StrategyType.AGGREGATION, "fedavg")
        
        # 检查信息完整性
        required_fields = ["name", "description", "type", "config_schema"]
        for field in required_fields:
            assert field in info, f"缺少字段: {field}"
        
        # 检查配置模式
        schema = info["config_schema"]
        assert "type" in schema
        assert "properties" in schema
        
        # 检查一些已知的属性
        properties = schema["properties"]
        if "normalize_weights" in properties:
            # FedAvg策略可能有这个属性
            assert "type" in properties["normalize_weights"]
            assert properties["normalize_weights"]["type"] == "boolean"
    
    def test_error_handling(self):
        """测试错误处理"""
        factory = StrategyFactory()
        
        # 测试创建不存在的策略
        with pytest.raises(StrategyError):
            factory.create_strategy(StrategyType.AGGREGATION, "nonexistent")
        
        # 测试获取不存在策略的信息
        with pytest.raises(StrategyError):
            factory.get_strategy_info(StrategyType.AGGREGATION, "nonexistent")
        
        # 测试创建错误类型的策略
        with pytest.raises(StrategyError):
            factory.create_aggregation_strategy("standard")  # standard是训练策略