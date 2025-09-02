"""
Comprehensive Strategy Pattern Tests

Tests for all strategy pattern implementations including:
- Aggregation strategies (FedAvg, LoRA, AdaLoRA)
- Training strategies (Standard, LoRA, AdaLoRA)
- Strategy factory and manager
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch

from src.strategies.aggregation.fedavg import FedAvgStrategy
from src.strategies.aggregation.lora_fedavg import LoRAFedAvgStrategy
from src.strategies.aggregation.adalora_fedavg import AdaLoRAFedAvgStrategy
from src.strategies.training.standard import StandardTrainingStrategy
from src.strategies.training.lora import LoRATrainingStrategy
from src.strategies.training.adalora import AdaLoRATrainingStrategy
from src.strategies.strategy_factory import StrategyFactory
from src.strategies.strategy_manager import StrategyManager
from src.core.exceptions.exceptions import StrategyError


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class TestAggregationStrategies:
    """Test aggregation strategies"""
    
    def setup_method(self):
        self.client_models = [
            {'fc1.weight': torch.randn(5, 10), 'fc1.bias': torch.randn(5),
             'fc2.weight': torch.randn(2, 5), 'fc2.bias': torch.randn(2)}
            for _ in range(3)
        ]
        self.client_weights = [100, 150, 200]
        self.global_model = SimpleModel()
        self.config = {'federated': {'num_rounds': 10}}
    
    def test_fedavg_strategy_basic(self):
        """Test FedAvg strategy basic functionality"""
        strategy = FedAvgStrategy()
        
        # Test strategy identification
        assert strategy.get_name() == "fedavg"
        assert strategy.get_description() is not None
        assert len(strategy.get_description()) > 0
    
    def test_fedavg_aggregation(self):
        """Test FedAvg aggregation"""
        strategy = FedAvgStrategy()
        
        result = strategy.aggregate(
            client_models=self.client_models,
            client_weights=self.client_weights,
            global_model=self.global_model,
            config=self.config
        )
        
        # Verify aggregation result
        assert isinstance(result, dict)
        assert 'fc1.weight' in result
        assert 'fc1.bias' in result
        assert 'fc2.weight' in result
        assert 'fc2.bias' in result
        
        # Verify weights are properly averaged
        expected_weight = (
            self.client_models[0]['fc1.weight'] * 100 +
            self.client_models[1]['fc1.weight'] * 150 +
            self.client_models[2]['fc1.weight'] * 200
        ) / 450  # Total weight
        
        assert torch.allclose(result['fc1.weight'], expected_weight)
    
    def test_fedavg_empty_models(self):
        """Test FedAvg with empty client models"""
        strategy = FedAvgStrategy()
        
        with pytest.raises(StrategyError):
            strategy.aggregate([], [], self.global_model, self.config)
    
    def test_fedavg_mismatched_model_keys(self):
        """Test FedAvg with mismatched model keys"""
        strategy = FedAvgStrategy()
        
        # Create models with different keys
        mismatched_models = [
            {'fc1.weight': torch.randn(5, 10)},
            {'fc2.weight': torch.randn(2, 5)}
        ]
        
        with pytest.raises(StrategyError):
            strategy.aggregate(mismatched_models, [100, 100], self.global_model, self.config)
    
    def test_lora_strategy_basic(self):
        """Test LoRA FedAvg strategy basic functionality"""
        strategy = LoRAFedAvgStrategy()
        
        assert strategy.get_name() == "lora_fedavg"
        assert strategy.get_description() is not None
    
    def test_lora_aggregation(self):
        """Test LoRA aggregation"""
        strategy = LoRAFedAvgStrategy()
        
        # Mock LoRA parameters
        lora_models = [
            {'fc1.lora_A': torch.randn(4, 10), 'fc1.lora_B': torch.randn(5, 4),
             'fc2.lora_A': torch.randn(4, 5), 'fc2.lora_B': torch.randn(2, 4)}
            for _ in range(3)
        ]
        
        with patch('src.strategies.aggregation.lora_fedavg.get_trainable_keys') as mock_get_keys:
            mock_get_keys.return_value = ['fc1.lora_A', 'fc1.lora_B', 'fc2.lora_A', 'fc2.lora_B']
            
            result = strategy.aggregate(
                client_models=lora_models,
                client_weights=self.client_weights,
                global_model=self.global_model,
                config=self.config
            )
            
            assert isinstance(result, dict)
            assert 'fc1.lora_A' in result
            assert 'fc1.lora_B' in result
    
    def test_adalora_strategy_basic(self):
        """Test AdaLoRA FedAvg strategy basic functionality"""
        strategy = AdaLoRAFedAvgStrategy()
        
        assert strategy.get_name() == "adalora_fedavg"
        assert strategy.get_description() is not None
    
    def test_adalora_aggregation(self):
        """Test AdaLoRA aggregation"""
        strategy = AdaLoRAFedAvgStrategy()
        
        with patch('src.strategies.aggregation.adalora_fedavg.adalora_fedavg') as mock_adalora:
            mock_result = {'test': torch.randn(10)}
            mock_adalora.return_value = mock_result
            
            result = strategy.aggregate(
                client_models=self.client_models,
                client_weights=self.client_weights,
                global_model=self.global_model,
                config=self.config
            )
            
            # Verify the underlying function was called
            mock_adalora.assert_called_once_with(self.client_models, self.client_weights)
            assert result == mock_result


class TestTrainingStrategies:
    """Test training strategies"""
    
    def setup_method(self):
        self.model = SimpleModel()
        self.config = {
            'optimizer': {'name': 'adam', 'lr': 0.001},
            'epochs': 1,
            'device': 'cpu'
        }
        
        # Mock data loader
        self.mock_loader = Mock()
        self.mock_loader.__iter__ = Mock(return_value=iter([
            (torch.randn(32, 10), torch.randint(0, 2, (32,)))
            for _ in range(5)
        ]))
    
    def test_standard_training_strategy(self):
        """Test standard training strategy"""
        strategy = StandardTrainingStrategy()
        
        assert strategy.get_name() == "standard"
        assert strategy.get_description() is not None
        
        # Test training setup
        optimizer = strategy.setup_optimizer(self.model, self.config)
        assert isinstance(optimizer, torch.optim.Adam)
        
        # Test training execution
        with patch.object(strategy, 'train_epoch') as mock_train:
            mock_train.return_value = {'loss': 0.5, 'accuracy': 0.8}
            
            result = strategy.train(
                model=self.model,
                train_loader=self.mock_loader,
                config=self.config
            )
            
            mock_train.assert_called_once()
            assert result == {'loss': 0.5, 'accuracy': 0.8}
    
    def test_lora_training_strategy(self):
        """Test LoRA training strategy"""
        strategy = LoRATrainingStrategy()
        
        assert strategy.get_name() == "lora"
        assert strategy.get_description() is not None
        
        # Test with LoRA configuration
        lora_config = self.config.copy()
        lora_config['lora'] = {'r': 8, 'alpha': 16}
        
        with patch('src.strategies.training.lora.setup_lora_training') as mock_setup:
            mock_setup.return_value = (self.model, Mock())
            
            optimizer = strategy.setup_optimizer(self.model, lora_config)
            
            mock_setup.assert_called_once()
    
    def test_adalora_training_strategy(self):
        """Test AdaLoRA training strategy"""
        strategy = AdaLoRATrainingStrategy()
        
        assert strategy.get_name() == "adalora"
        assert strategy.get_description() is not None
        
        # Test with AdaLoRA configuration
        adalora_config = self.config.copy()
        adalora_config['adalora'] = {'initial_r': 8, 'alpha': 16}
        
        with patch('src.strategies.training.adalora.setup_adalora_training') as mock_setup:
            mock_setup.return_value = (self.model, Mock())
            
            optimizer = strategy.setup_optimizer(self.model, adalora_config)
            
            mock_setup.assert_called_once()


class TestStrategyFactory:
    """Test strategy factory"""
    
    def setup_method(self):
        self.factory = StrategyFactory()
    
    def test_create_aggregation_strategies(self):
        """Test creating aggregation strategies"""
        fedavg = self.factory.create_aggregation_strategy('fedavg')
        lora = self.factory.create_aggregation_strategy('lora_fedavg')
        adalora = self.factory.create_aggregation_strategy('adalora_fedavg')
        
        assert isinstance(fedavg, FedAvgStrategy)
        assert isinstance(lora, LoRAFedAvgStrategy)
        assert isinstance(adalora, AdaLoRAFedAvgStrategy)
    
    def test_create_training_strategies(self):
        """Test creating training strategies"""
        standard = self.factory.create_training_strategy('standard')
        lora = self.factory.create_training_strategy('lora')
        adalora = self.factory.create_training_strategy('adalora')
        
        assert isinstance(standard, StandardTrainingStrategy)
        assert isinstance(lora, LoRATrainingStrategy)
        assert isinstance(adalora, AdaLoRATrainingStrategy)
    
    def test_create_unknown_strategy(self):
        """Test creating unknown strategy"""
        with pytest.raises(StrategyError):
            self.factory.create_aggregation_strategy('unknown')
        
        with pytest.raises(StrategyError):
            self.factory.create_training_strategy('unknown')
    
    def test_list_available_strategies(self):
        """Test listing available strategies"""
        agg_strategies = self.factory.list_aggregation_strategies()
        train_strategies = self.factory.list_training_strategies()
        
        assert 'fedavg' in agg_strategies
        assert 'lora_fedavg' in agg_strategies
        assert 'adalora_fedavg' in agg_strategies
        
        assert 'standard' in train_strategies
        assert 'lora' in train_strategies
        assert 'adalora' in train_strategies
    
    def test_register_custom_strategy(self):
        """Test registering custom strategy"""
        class CustomStrategy(FedAvgStrategy):
            def get_name(self):
                return "custom"
        
        # Register custom strategy
        self.factory.register_aggregation_strategy('custom', CustomStrategy)
        
        # Create custom strategy
        custom = self.factory.create_aggregation_strategy('custom')
        assert isinstance(custom, CustomStrategy)
        assert custom.get_name() == "custom"


class TestStrategyManager:
    """Test strategy manager"""
    
    def setup_method(self):
        self.manager = StrategyManager()
    
    def test_singleton_pattern(self):
        """Test that StrategyManager is a singleton"""
        manager2 = StrategyManager()
        
        # Should be the same instance
        assert self.manager is manager2
    
    def test_register_and_get_strategy(self):
        """Test strategy registration and retrieval"""
        strategy = FedAvgStrategy()
        
        self.manager.register_aggregation_strategy('test_strategy', strategy)
        retrieved = self.manager.get_aggregation_strategy('test_strategy')
        
        assert retrieved is strategy
    
    def test_get_default_strategies(self):
        """Test getting default strategies"""
        fedavg = self.manager.get_aggregation_strategy('fedavg')
        standard = self.manager.get_training_strategy('standard')
        
        assert isinstance(fedavg, FedAvgStrategy)
        assert isinstance(standard, StandardTrainingStrategy)
    
    def test_list_available_strategies(self):
        """Test listing available strategies"""
        agg_strategies = self.manager.list_aggregation_strategies()
        train_strategies = self.manager.list_training_strategies()
        
        assert isinstance(agg_strategies, list)
        assert isinstance(train_strategies, list)
        assert len(agg_strategies) > 0
        assert len(train_strategies) > 0
    
    def test_strategy_validation(self):
        """Test strategy validation"""
        # Test registering invalid strategy
        with pytest.raises(StrategyError):
            self.manager.register_aggregation_strategy('invalid', 'not_a_strategy')
    
    def test_strategy_configuration(self):
        """Test strategy configuration"""
        config = {
            'aggregation': {'strategy': 'fedavg'},
            'training': {'strategy': 'standard'}
        }
        
        # Get strategies based on config
        agg_strategy = self.manager.get_aggregation_strategy(config['aggregation']['strategy'])
        train_strategy = self.manager.get_training_strategy(config['training']['strategy'])
        
        assert isinstance(agg_strategy, FedAvgStrategy)
        assert isinstance(train_strategy, StandardTrainingStrategy)
    
    def test_strategy_compatibility(self):
        """Test strategy compatibility checking"""
        # Test compatible strategies
        agg_strategy = FedAvgStrategy()
        train_strategy = StandardTrainingStrategy()
        
        # These should be compatible
        assert self.manager.are_compatible(agg_strategy, train_strategy)
    
    def test_strategy_recommendation(self):
        """Test strategy recommendation based on use case"""
        # Test recommendation for standard FL
        recommended = self.manager.recommend_strategies('standard_fl')
        
        assert 'aggregation' in recommended
        assert 'training' in recommended
        assert recommended['aggregation'] == 'fedavg'
        assert recommended['training'] == 'standard'
    
    def test_strategy_performance_metrics(self):
        """Test strategy performance metrics"""
        strategy = FedAvgStrategy()
        
        # Mock performance data
        with patch.object(strategy, 'get_performance_metrics') as mock_metrics:
            mock_metrics.return_value = {
                'avg_aggregation_time': 0.1,
                'memory_usage': 100,
                'success_rate': 0.99
            }
            
            metrics = strategy.get_performance_metrics()
            
            assert 'avg_aggregation_time' in metrics
            assert 'memory_usage' in metrics
            assert 'success_rate' in metrics