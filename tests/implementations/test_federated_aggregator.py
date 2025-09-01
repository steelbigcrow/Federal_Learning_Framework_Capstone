"""
Unit tests for FederatedAggregator implementation.
"""

import pytest
import torch
import torch.nn as nn
from typing import Dict, List, Set
from unittest.mock import Mock, patch

from src.implementations.aggregators.federated_aggregator import FederatedAggregator
from src.core.base.aggregator import AbstractAggregator, AggregationMode
from src.core.exceptions.exceptions import AggregationError


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)
    
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class TestFederatedAggregator:
    """Test suite for FederatedAggregator class."""
    
    @pytest.fixture
    def client_models(self):
        return [
            {
                'fc1.weight': torch.randn(5, 10),
                'fc1.bias': torch.randn(5),
                'fc2.weight': torch.randn(2, 5),
                'fc2.bias': torch.randn(2)
            }
            for _ in range(3)
        ]
    
    @pytest.fixture
    def client_weights(self):
        return [100, 150, 200]
    
    @pytest.fixture
    def aggregator(self):
        return FederatedAggregator()
    
    def test_inheritance(self, aggregator):
        assert isinstance(aggregator, AbstractAggregator)
    
    def test_initialization_default(self, aggregator):
        assert aggregator._aggregation_mode == AggregationMode.WEIGHTED_AVERAGE
        assert aggregator._config == {}
        assert aggregator._parameter_filter is None
    
    def test_aggregate_weighted_average(self, aggregator, client_models, client_weights):
        result = aggregator.aggregate(client_models, client_weights)
        
        assert isinstance(result, dict)
        assert set(result.keys()) == set(client_models[0].keys())
        
        for key, tensor in result.items():
            assert isinstance(tensor, torch.Tensor)
    
    def test_aggregate_error_handling(self, aggregator):
        with pytest.raises(AggregationError) as exc_info:
            aggregator.aggregate([], [])
        
        assert "Aggregation failed" in str(exc_info.value)
    
    def test_compute_aggregation_weights_normal(self, client_weights):
        aggregator = FederatedAggregator()
        result = aggregator.compute_aggregation_weights(client_weights)
        
        assert abs(sum(result) - 1.0) < 1e-6
        assert len(result) == len(client_weights)
        
        expected = [w/sum(client_weights) for w in client_weights]
        for r, e in zip(result, expected):
            assert abs(r - e) < 1e-6
    
    def test_compute_aggregation_weights_zero_total(self):
        aggregator = FederatedAggregator()
        weights = [0, 0, 0]
        result = aggregator.compute_aggregation_weights(weights)
        
        expected = [1.0/3] * 3
        assert result == expected
    
    def test_get_trainable_keys(self, aggregator, client_models):
        result = aggregator.get_trainable_keys(client_models[0])
        
        assert isinstance(result, set)
        assert len(result) > 0
        assert set(result) == set(client_models[0].keys())
    
    def test_set_parameter_filter(self, aggregator):
        filter_set = {'fc1.weight', 'fc1.bias'}
        aggregator.set_parameter_filter(filter_set)
        
        assert aggregator._parameter_filter == filter_set
        assert aggregator._trainable_keys_cache is None
    
    def test_weighted_average_basic(self, aggregator, client_models, client_weights):
        result = aggregator._weighted_average(client_models, client_weights)
        
        assert set(result.keys()) == set(client_models[0].keys())
        
        total_weight = sum(client_weights)
        for key in result.keys():
            expected = sum(model[key] * (weight / total_weight) 
                          for model, weight in zip(client_models, client_weights))
            assert torch.allclose(result[key], expected, atol=1e-6)
    
    def test_backward_compatibility_fedavg(self, client_models, client_weights):
        result = FederatedAggregator.fedavg(client_models, client_weights)
        
        aggregator = FederatedAggregator()
        expected = aggregator._weighted_average(client_models, client_weights)
        
        assert set(result.keys()) == set(expected.keys())
        for key in result.keys():
            assert torch.allclose(result[key], expected[key], atol=1e-6)
    
    def test_backward_compatibility_get_trainable_keys_from_model(self):
        model = SimpleModel()
        result = FederatedAggregator.get_trainable_keys_from_model(model)
        
        expected = set(name for name, param in model.named_parameters() if param.requires_grad)
        assert result == expected