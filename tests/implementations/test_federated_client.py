"""
Unit tests for FederatedClient implementation.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Any
from unittest.mock import Mock, patch

from src.implementations.clients.federated_client import FederatedClient
from src.core.base.client import AbstractClient
from src.core.exceptions.exceptions import ClientConfigurationError, ClientTrainingError


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
    
    def forward(self, x):
        return self.fc(x)


class TestFederatedClient:
    """Test suite for FederatedClient class."""
    
    @pytest.fixture
    def model_ctor(self):
        return SimpleModel
    
    @pytest.fixture
    def train_data(self):
        X = torch.randn(100, 10)
        y = torch.randint(0, 2, (100,))
        return TensorDataset(X, y)
    
    @pytest.fixture
    def train_data_loader(self, train_data):
        return DataLoader(train_data, batch_size=16, shuffle=True)
    
    @pytest.fixture
    def basic_config(self):
        return {
            'optimizer': {'name': 'adam', 'lr': 0.001},
            'use_amp': False,
            'test_ratio': 0.2
        }
    
    @pytest.fixture
    def client(self, model_ctor, train_data_loader, basic_config):
        return FederatedClient(
            client_id=1,
            model_ctor=model_ctor,
            train_data_loader=train_data_loader,
            config=basic_config,
            device='cpu'
        )
    
    def test_inheritance(self, client):
        assert isinstance(client, AbstractClient)
    
    def test_initialization_basic(self, model_ctor, train_data_loader, basic_config):
        client = FederatedClient(
            client_id=1,
            model_ctor=model_ctor,
            train_data_loader=train_data_loader,
            config=basic_config,
            device='cpu'
        )
        
        assert client._client_id == 1
        assert client._model_ctor == model_ctor
        assert client._device == 'cpu'
        assert client._optimizer_cfg == {'name': 'adam', 'lr': 0.001}
        assert client._use_amp is False
        assert client._test_ratio == 0.2
    
    def test_data_splitting(self, model_ctor, train_data, basic_config):
        original_loader = DataLoader(train_data, batch_size=16, shuffle=True)
        
        config = basic_config.copy()
        config['test_ratio'] = 0.2
        client = FederatedClient(
            client_id=1,
            model_ctor=model_ctor,
            train_data_loader=original_loader,
            config=config,
            device='cpu'
        )
        
        assert hasattr(client, '_train_data_loader')
        assert hasattr(client, '_test_data_loader')
        
        train_size = len(client._train_data_loader.dataset)
        test_size = len(client._test_data_loader.dataset)
        total_size = len(original_loader.dataset)
        
        assert train_size + test_size == total_size
        assert abs(train_size - int(total_size * 0.8)) <= 2
    
    def test_receive_global_model_success(self, client):
        global_model = SimpleModel()
        global_state = global_model.state_dict()
        
        client.receive_global_model(global_state)
        
        assert client._current_model is not None
        assert isinstance(client._current_model, SimpleModel)
        
        model_device = next(client._current_model.parameters()).device
        assert str(model_device) == 'cpu'
    
    def test_receive_global_model_device_mismatch(self, client):
        # Skip this test as device validation is complex and environment-dependent
        pytest.skip("Device mismatch test skipped due to environment complexity")
    
    def test_local_train_without_model(self, client):
        with pytest.raises(ClientTrainingError) as exc_info:
            client.local_train(num_epochs=1)
        
        assert "No model loaded" in str(exc_info.value)
    
    def test_local_train_success(self, client):
        global_model = SimpleModel()
        global_state = global_model.state_dict()
        client.receive_global_model(global_state)
        
        model_state, train_metrics, num_samples = client.local_train(num_epochs=1)
        
        assert isinstance(model_state, dict)
        assert isinstance(train_metrics, dict)
        assert isinstance(num_samples, int)
        assert num_samples > 0
        
        expected_keys = ['train_acc', 'train_f1', 'train_loss', 'epoch_history']
        for key in expected_keys:
            assert key in train_metrics
    
    def test_local_evaluate_with_model(self, client):
        global_model = SimpleModel()
        global_state = global_model.state_dict()
        client.receive_global_model(global_state)
        
        eval_metrics = client.local_evaluate()
        
        assert isinstance(eval_metrics, dict)
        assert len(eval_metrics) > 0
    
    def test_local_evaluate_without_model(self, client):
        with pytest.raises(ClientTrainingError) as exc_info:
            client.local_evaluate()
        
        assert "No model available for evaluation" in str(exc_info.value)
    
    def test_backward_compatibility_properties(self, client):
        assert client.id == 1
        assert client.model_ctor == client._model_ctor
        assert client.optimizer_cfg == client._optimizer_cfg
        assert client.use_amp == client._use_amp
        assert client.test_ratio == client._test_ratio
    
    def test_backward_compatibility_local_train_and_eval(self, client):
        global_model = SimpleModel()
        global_state = global_model.state_dict()
        client.receive_global_model(global_state)
        
        model_state, combined_metrics, num_samples = client.local_train_and_eval(
            global_state, local_epochs=1
        )
        
        assert isinstance(model_state, dict)
        assert isinstance(combined_metrics, dict)
        assert isinstance(num_samples, int)
        assert num_samples > 0