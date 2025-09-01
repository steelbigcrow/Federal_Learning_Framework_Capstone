"""
Integration tests for Phase 4 implementations.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Any
import tempfile
import shutil

from src.implementations.clients.federated_client import FederatedClient
from src.implementations.servers.federated_server import FederatedServer
from src.implementations.aggregators.federated_aggregator import FederatedAggregator
from src.utils.paths import PathManager


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
    
    def forward(self, x):
        return self.fc(x)


class MockClient:
    def __init__(self, client_id: int, num_samples: int = 50):
        self.client_id = client_id
        self.num_samples = num_samples
        self.num_train_samples = num_samples


class TestIntegration:
    """Integration tests for Phase 4 implementations."""
    
    @pytest.fixture
    def temp_dir(self):
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def path_manager(self, temp_dir):
        return PathManager(root=temp_dir, dataset_name="integration", model_name="test", use_lora=False)
    
    @pytest.fixture
    def model_constructor(self):
        return SimpleModel
    
    @pytest.fixture
    def train_data(self):
        X = torch.randn(100, 10)
        y = torch.randint(0, 2, (100,))
        return TensorDataset(X, y)
    
    @pytest.fixture
    def mock_clients(self, train_data):
        clients = []
        for i in range(3):
            client_data = torch.utils.data.Subset(train_data, list(range(i*30, (i+1)*30)))
            loader = DataLoader(client_data, batch_size=16, shuffle=True)
            clients.append(loader)
        return clients
    
    @pytest.fixture
    def federated_clients(self, model_constructor, mock_clients):
        clients = []
        for i, loader in enumerate(mock_clients):
            client = FederatedClient(
                client_id=i,
                model_ctor=model_constructor,
                train_data_loader=loader,
                config={
                    'optimizer': {'name': 'adam', 'lr': 0.001},
                    'use_amp': False,
                    'test_ratio': 0.2
                },
                device='cpu'
            )
            clients.append(client)
        return clients
    
    @pytest.fixture
    def federated_server(self, model_constructor, federated_clients, path_manager):
        mock_clients = [MockClient(i) for i in range(3)]
        return FederatedServer(
            model_constructor=model_constructor,
            clients=mock_clients,
            path_manager=path_manager,
            config={
                'federated': {'num_rounds': 1},
                'lora_cfg': {},
                'adalora_cfg': {},
                'save_client_each_round': False,
                'model_info': {'dataset': 'test', 'model_type': 'simple'}
            },
            device='cpu'
        )
    
    @pytest.fixture
    def federated_aggregator(self):
        return FederatedAggregator()
    
    def test_end_to_end_workflow(self, federated_server, federated_clients, federated_aggregator):
        """Test complete end-to-end federated learning workflow."""
        # Initialize global model
        federated_server.initialize_global_model()
        assert federated_server._global_model is not None
        
        # Distribute global model to clients
        global_state = federated_server._global_model.state_dict()
        for client in federated_clients:
            client.receive_global_model(global_state)
            assert client._current_model is not None
        
        # Local training on clients
        client_models = []
        client_weights = []
        
        for client in federated_clients:
            model_state, train_metrics, num_samples = client.local_train(num_epochs=1)
            client_models.append(model_state)
            client_weights.append(num_samples)
        
        assert len(client_models) == len(federated_clients)
        assert len(client_weights) == len(federated_clients)
        
        # Aggregate models
        aggregated_state = federated_aggregator.aggregate(client_models, client_weights)
        assert isinstance(aggregated_state, dict)
        
        # Update global model
        federated_server._global_model.load_state_dict(aggregated_state)
    
    def test_client_server_integration(self, federated_server, federated_clients):
        """Test client-server integration."""
        federated_server.initialize_global_model()
        
        # Test client selection
        selected = federated_server.select_clients(round_number=1)
        assert len(selected) > 0
        
        # Test model distribution with real FederatedClient instances
        global_state = federated_server._global_model.state_dict()
        for client in federated_clients:
            client.receive_global_model(global_state)
            assert client._current_model is not None
    
    def test_aggregator_integration(self, federated_clients, federated_aggregator):
        """Test aggregator integration with federated clients."""
        # Get model states from clients
        client_models = []
        client_weights = []
        
        for client in federated_clients:
            global_model = SimpleModel()
            global_state = global_model.state_dict()
            client.receive_global_model(global_state)
            
            model_state, _, num_samples = client.local_train(num_epochs=1)
            client_models.append(model_state)
            client_weights.append(num_samples)
        
        # Aggregate using federated aggregator
        aggregated_state = federated_aggregator.aggregate(client_models, client_weights)
        
        # Verify aggregation result
        assert isinstance(aggregated_state, dict)
        assert len(aggregated_state) > 0
        
        # Verify aggregated model can be loaded
        test_model = SimpleModel()
        test_model.load_state_dict(aggregated_state, strict=False)
        assert test_model is not None
    
    def test_backward_compatibility(self, model_constructor, path_manager):
        """Test backward compatibility across components."""
        # Create old-style mock clients
        old_clients = [MockClient(i) for i in range(3)]
        
        # Create server with old-style clients
        server = FederatedServer(
            model_constructor=model_constructor,
            clients=old_clients,
            path_manager=path_manager,
            config={'federated': {'num_rounds': 1}, 'model_info': {}},
            device='cpu'
        )
        
        # Test backward compatibility properties
        assert server.model_ctor == model_constructor
        assert server.clients == old_clients
        assert len(server.clients) == 3