"""
Unit tests for FederatedServer implementation.
"""

import pytest
import torch
import torch.nn as nn
from typing import Dict, Any
from unittest.mock import Mock, patch
import tempfile
import shutil

from src.implementations.servers.federated_server import FederatedServer
from src.core.base.server import AbstractServer
from src.core.exceptions.exceptions import ServerConfigurationError, ServerOperationError
from src.utils.paths import PathManager


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
    
    def forward(self, x):
        return self.fc(x)


class MockClient:
    def __init__(self, client_id: int):
        self.client_id = client_id
        self.num_samples = 100
        self.num_train_samples = 100


class TestFederatedServer:
    """Test suite for FederatedServer class."""
    
    @pytest.fixture
    def temp_dir(self):
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def path_manager(self, temp_dir):
        return PathManager(root=temp_dir, dataset_name="test", model_name="test", use_lora=False)
    
    @pytest.fixture
    def model_constructor(self):
        return SimpleModel
    
    @pytest.fixture
    def clients(self):
        return [MockClient(i) for i in range(3)]
    
    @pytest.fixture
    def basic_config(self):
        return {
            'federated': {
                'num_rounds': 1
            },
            'lora_cfg': {},
            'adalora_cfg': {},
            'save_client_each_round': False,
            'model_info': {'dataset': 'test', 'model_type': 'simple'}
        }
    
    @pytest.fixture
    def server(self, model_constructor, clients, path_manager, basic_config):
        return FederatedServer(
            model_constructor=model_constructor,
            clients=clients,
            path_manager=path_manager,
            config=basic_config,
            device='cpu'
        )
    
    def test_inheritance(self, server):
        assert isinstance(server, AbstractServer)
    
    def test_initialization_basic(self, model_constructor, clients, path_manager, basic_config):
        server = FederatedServer(
            model_constructor=model_constructor,
            clients=clients,
            path_manager=path_manager,
            config=basic_config,
            device='cpu'
        )
        
        assert server._model_constructor == model_constructor
        assert server._clients == clients
        assert server._device == 'cpu'
        assert server._path_manager == path_manager
        assert server._save_client_each_round is False
    
    def test_initialize_global_model_success(self, server):
        server.initialize_global_model()
        
        assert server._global_model is not None
        assert isinstance(server._global_model, SimpleModel)
        
        model_device = next(server._global_model.parameters()).device
        assert str(model_device) == 'cpu'
    
    def test_select_clients_default(self, server):
        selected = server.select_clients(round_number=1)
        
        assert len(selected) == 3
        assert selected is not server._clients  # Check it's not the same object
        assert all(client in server._clients for client in selected)
    
    def test_aggregate_models_basic(self, server):
        server.initialize_global_model()
        
        client_models = [
            {'fc.weight': torch.randn(2, 10), 'fc.bias': torch.randn(2)}
            for _ in range(3)
        ]
        client_weights = [100, 150, 200]
        
        result = server.aggregate_models(client_models, client_weights)
        
        assert isinstance(result, dict)
        assert len(result) > 0
    
    def test_aggregate_models_error_handling(self, server):
        server.initialize_global_model()
        
        with pytest.raises(ServerOperationError) as exc_info:
            server.aggregate_models([], [])
        
        assert "Model aggregation failed" in str(exc_info.value)
    
    def test_save_checkpoint_basic(self, server):
        server.initialize_global_model()
        
        with patch.object(server, '_save_client_checkpoints'), \
             patch.object(server, '_save_global_checkpoint'), \
             patch.object(server, '_save_round_metrics'), \
             patch.object(server, '_generate_metrics_plots'):
            
            server.save_checkpoint(round_number=1, metrics={'loss': 0.5})
    
    def test_backward_compatibility_properties(self, server):
        assert server.model_ctor == server._model_constructor
        assert server.clients == server._clients
        assert server.paths == server._path_manager
        assert server.lora_cfg == server._lora_cfg
        assert server.save_client_each_round == server._save_client_each_round
        assert server.model_info == server._model_info
    
    def test_backward_compatibility_run_method(self, server):
        with patch.object(server, 'run_federated_learning', return_value={'success': True}):
            with patch('builtins.print'):
                server.run(num_rounds=1, local_epochs=1)
                
                server.run_federated_learning.assert_called_once_with(1, 1)