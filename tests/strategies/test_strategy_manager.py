"""
策略管理器单元测试
"""

import pytest
import json
from unittest.mock import Mock, patch
from typing import Dict, Any

from src.strategies.strategy_manager import StrategyManager, StrategyConfiguration, StrategyPerformanceMetrics
from src.core.interfaces.strategy import StrategyType
from src.core.exceptions.exceptions import StrategyError


class TestStrategyManager:
    """测试策略管理器"""
    
    def setup_method(self):
        """测试前设置"""
        self.manager = StrategyManager()
        
    def test_initialization(self):
        """测试初始化"""
        assert self.manager.factory is not None
        assert isinstance(self.manager.performance_metrics, dict)
        assert isinstance(self.manager.strategy_configs, dict)
        assert isinstance(self.manager.execution_history, list)
    
    def test_register_strategy_config(self):
        """测试注册策略配置"""
        config = {"lr": 0.001, "epochs": 10}
        
        self.manager.register_strategy_config(
            strategy_type=StrategyType.TRAINING,
            strategy_name="standard",
            config=config,
            enabled=True,
            priority=1,
            description="标准训练配置"
        )
        
        key = "training:standard"
        assert key in self.manager.strategy_configs
        
        stored_config = self.manager.strategy_configs[key]
        assert stored_config.strategy_type == StrategyType.TRAINING
        assert stored_config.strategy_name == "standard"
        assert stored_config.config == config
        assert stored_config.enabled == True
        assert stored_config.priority == 1
        assert stored_config.description == "标准训练配置"
    
    def test_get_strategy_config(self):
        """测试获取策略配置"""
        config = {"lr": 0.001}
        self.manager.register_strategy_config(
            StrategyType.AGGREGATION, "fedavg", config
        )
        
        result = self.manager.get_strategy_config(StrategyType.AGGREGATION, "fedavg")
        assert result is not None
        assert result.config == config
        
        # 测试不存在的配置
        result = self.manager.get_strategy_config(StrategyType.TRAINING, "nonexistent")
        assert result is None
    
    def test_get_strategy_key(self):
        """测试获取策略键"""
        key = self.manager._get_strategy_key(StrategyType.AGGREGATION, "fedavg")
        assert key == "aggregation:fedavg"
    
    @patch('src.strategies.strategy_manager.StrategyFactory')
    def test_execute_strategy_success(self, mock_factory_class):
        """测试成功执行策略"""
        # 创建模拟的策略和工厂
        mock_strategy = Mock()
        mock_strategy.execute.return_value = "success_result"
        
        mock_factory = Mock()
        mock_factory.create_strategy.return_value = mock_strategy
        mock_factory.get_available_strategies.return_value = ["fedavg"]
        mock_factory_class.return_value = mock_factory
        
        manager = StrategyManager()
        manager.factory = mock_factory
        
        context = {"data": "test"}
        result = manager.execute_strategy(
            StrategyType.AGGREGATION, "fedavg", context
        )
        
        assert result == "success_result"
        mock_strategy.execute.assert_called_once_with(context)
        
        # 检查性能指标
        metrics = manager.get_performance_metrics(StrategyType.AGGREGATION, "fedavg")
        assert metrics is not None
        assert metrics.execution_count == 1
        assert metrics.success_count == 1
        assert metrics.error_count == 0
    
    @patch('src.strategies.strategy_manager.StrategyFactory')
    def test_execute_strategy_error(self, mock_factory_class):
        """测试策略执行失败"""
        mock_strategy = Mock()
        mock_strategy.execute.side_effect = StrategyError("Test error")
        
        mock_factory = Mock()
        mock_factory.create_strategy.return_value = mock_strategy
        mock_factory.get_available_strategies.return_value = ["fedavg"]
        mock_factory_class.return_value = mock_factory
        
        manager = StrategyManager()
        manager.factory = mock_factory
        
        context = {"data": "test"}
        
        with pytest.raises(StrategyError, match="Test error"):
            manager.execute_strategy(StrategyType.AGGREGATION, "fedavg", context)
        
        # 检查性能指标
        metrics = manager.get_performance_metrics(StrategyType.AGGREGATION, "fedavg")
        assert metrics is not None
        assert metrics.execution_count == 1
        assert metrics.success_count == 0
        assert metrics.error_count == 1
        assert "Test error" in metrics.last_error
    
    def test_get_performance_metrics(self):
        """测试获取性能指标"""
        metrics = self.manager.get_performance_metrics(StrategyType.AGGREGATION, "fedavg")
        assert metrics is not None
        assert isinstance(metrics, StrategyPerformanceMetrics)
        assert metrics.execution_count == 0
    
    def test_get_all_performance_metrics(self):
        """测试获取所有性能指标"""
        all_metrics = self.manager.get_all_performance_metrics()
        assert isinstance(all_metrics, dict)
        assert len(all_metrics) > 0
    
    def test_get_execution_history(self):
        """测试获取执行历史"""
        history = self.manager.get_execution_history()
        assert isinstance(history, list)
        assert len(history) == 0
    
    def test_recommend_strategy(self):
        """测试推荐策略"""
        # 初始状态下应该返回默认策略
        recommended = self.manager.recommend_strategy(StrategyType.AGGREGATION, {})
        assert recommended == "fedavg"
    
    def test_export_import_configurations(self):
        """测试导出导入配置"""
        config = {"lr": 0.001, "batch_size": 32}
        self.manager.register_strategy_config(
            StrategyType.TRAINING, "standard", config, True, 1, "测试配置"
        )
        
        # 导出配置
        exported = self.manager.export_configurations()
        assert isinstance(exported, str)
        
        # 清空当前配置
        self.manager.strategy_configs.clear()
        
        # 导入配置
        self.manager.import_configurations(exported)
        
        # 验证配置恢复
        imported_config = self.manager.get_strategy_config(StrategyType.TRAINING, "standard")
        assert imported_config is not None
        assert imported_config.config == config
        assert imported_config.enabled == True
        assert imported_config.priority == 1
        assert imported_config.description == "测试配置"
    
    def test_clear_history(self):
        """测试清空历史"""
        # 添加一些历史记录
        self.manager.execution_history.extend([{"test": "data"}] * 5)
        
        self.manager.clear_history()
        assert len(self.manager.execution_history) == 0
    
    def test_reset_performance_metrics(self):
        """测试重置性能指标"""
        # 修改一些指标
        key = "aggregation:fedavg"
        original_metrics = self.manager.performance_metrics[key].execution_count
        self.manager.performance_metrics[key].execution_count = 100
        
        self.manager.reset_performance_metrics()
        
        # 检查是否重置
        assert self.manager.performance_metrics[key].execution_count == original_metrics


class TestStrategyPerformanceMetrics:
    """测试策略性能指标"""
    
    def test_average_execution_time(self):
        """测试平均执行时间计算"""
        metrics = StrategyPerformanceMetrics()
        assert metrics.average_execution_time == 0.0
        
        metrics.execution_count = 5
        metrics.total_execution_time = 10.0
        assert metrics.average_execution_time == 2.0
    
    def test_success_rate(self):
        """测试成功率计算"""
        metrics = StrategyPerformanceMetrics()
        assert metrics.success_rate == 0.0
        
        metrics.execution_count = 10
        metrics.success_count = 8
        assert metrics.success_rate == 0.8
        
        metrics.error_count = 2
        assert metrics.success_count + metrics.error_count == metrics.execution_count


class TestStrategyConfiguration:
    """测试策略配置"""
    
    def test_strategy_configuration(self):
        """测试策略配置数据结构"""
        config = StrategyConfiguration(
            strategy_type=StrategyType.TRAINING,
            strategy_name="standard",
            config={"lr": 0.001},
            enabled=True,
            priority=1,
            description="测试配置"
        )
        
        assert config.strategy_type == StrategyType.TRAINING
        assert config.strategy_name == "standard"
        assert config.config == {"lr": 0.001}
        assert config.enabled == True
        assert config.priority == 1
        assert config.description == "测试配置"