"""
Comprehensive OOP Integration Tests

This test suite verifies that the OOP refactored code can run independently
and provides all the functionality of the original architecture.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch
import tempfile
import os
from typing import Dict, List, Any

from src.implementations.servers import FederatedServer
from src.implementations.clients import FederatedClient
from src.implementations.aggregators import FederatedAggregator
from src.strategies.aggregation import FedAvgStrategy, LoRAFedAvgStrategy, AdaLoRAFedAvgStrategy
from src.utils.paths import PathManager
from src.core.exceptions.exceptions import ServerConfigurationError, ServerOperationError


class SimpleModel(nn.Module):
    """Simple test model"""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
    
    def forward(self, x):
        return self.fc(x)


class TestOOPIntegration:
    """Test suite for OOP refactored code integration"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.path_manager = PathManager(
            root=self.temp_dir,
            dataset_name="test",
            model_name="simple",
            use_lora=False,
            use_adalora=False
        )
        
        # Mock data loader
        self.mock_loader = Mock()
        self.mock_loader.__iter__ = Mock(return_value=iter([
            (torch.randn(32, 10), torch.randint(0, 2, (32,)))
            for _ in range(5)
        ]))
        
        # Model constructor
        self.model_constructor = lambda: SimpleModel()
        
        # Mock clients
        self.clients = [
            Mock(client_id=i, train_loader=self.mock_loader, num_samples=100)
            for i in range(3)
        ]
        
        # Basic configuration
        self.basic_config = {
            'federated': {
                'num_rounds': 10
            },
            'lora_cfg': {},
            'adalora_cfg': {},
            'save_client_each_round': True,
            'model_info': {
                'dataset': 'test',
                'model_type': 'simple'
            }
        }
    
    def test_federated_server_initialization(self):
        """Test FederatedServer can be initialized properly"""
        server = FederatedServer(
            model_constructor=self.model_constructor,
            clients=self.clients,
            path_manager=self.path_manager,
            config=self.basic_config,
            device="cpu"
        )
        
        assert server is not None
        assert server.model_info == self.basic_config['model_info']
        assert server.lora_cfg == {}
        assert server.adalora_cfg == {}
        assert server.save_client_each_round == True
    
    def test_federated_server_with_lora_config(self):
        """Test FederatedServer with LoRA configuration"""
        lora_config = self.basic_config.copy()
        lora_config['lora_cfg'] = {
            'r': 8,
            'alpha': 16,
            'replaced_modules': ['fc']
        }
        
        server = FederatedServer(
            model_constructor=self.model_constructor,
            clients=self.clients,
            path_manager=self.path_manager,
            config=lora_config,
            device="cpu"
        )
        
        assert server.lora_cfg['r'] == 8
        assert server.lora_cfg['replaced_modules'] == ['fc']
    
    def test_federated_server_with_adalora_config(self):
        """Test FederatedServer with AdaLoRA configuration"""
        adalora_config = self.basic_config.copy()
        adalora_config['adalora_cfg'] = {
            'initial_r': 8,
            'alpha': 16,
            'replaced_modules': ['fc']
        }
        
        server = FederatedServer(
            model_constructor=self.model_constructor,
            clients=self.clients,
            path_manager=self.path_manager,
            config=adalora_config,
            device="cpu"
        )
        
        assert server.adalora_cfg['initial_r'] == 8
        assert server.adalora_cfg['replaced_modules'] == ['fc']
    
    def test_federated_client_initialization(self):
        """Test FederatedClient can be initialized properly"""
        # Create a simple mock dataset for the loader
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=100)
        self.mock_loader.dataset = mock_dataset
        
        client = FederatedClient(
            client_id=0,
            model_ctor=self.model_constructor,
            train_data_loader=self.mock_loader,
            config={'optimizer': {'name': 'adam', 'lr': 0.001}},
            device="cpu"
        )
        
        assert client is not None
        assert client.id() == 0
        assert client.use_amp() == False
    
    def test_aggregation_strategies_availability(self):
        """Test all aggregation strategies are available"""
        # Test FedAvg strategy
        fedavg_strategy = FedAvgStrategy()
        assert fedavg_strategy.get_name() == "fedavg"
        
        # Test LoRA strategy
        lora_strategy = LoRAFedAvgStrategy()
        assert lora_strategy.get_name() == "lora_fedavg"
        
        # Test AdaLoRA strategy
        adalora_strategy = AdaLoRAFedAvgStrategy()
        assert adalora_strategy.get_name() == "adalora_fedavg"
    
    def test_server_with_strategy_pattern(self):
        """Test server with strategy pattern aggregation"""
        strategy = FedAvgStrategy()
        server = FederatedServer(
            model_constructor=self.model_constructor,
            clients=self.clients,
            path_manager=self.path_manager,
            config=self.basic_config,
            device="cpu",
            aggregation_strategy=strategy
        )
        
        # Verify strategy is set
        assert server._aggregation_strategy is not None
        assert server._aggregation_strategy.get_name() == "fedavg"
    
    def test_default_aggregation_logic_standard(self):
        """Test default aggregation logic for standard mode"""
        server = FederatedServer(
            model_constructor=self.model_constructor,
            clients=self.clients,
            path_manager=self.path_manager,
            config=self.basic_config,
            device="cpu"
        )
        
        # Mock client models
        client_models = [
            {'fc.weight': torch.randn(2, 10), 'fc.bias': torch.randn(2)}
            for _ in range(3)
        ]
        client_weights = [100, 100, 100]
        
        # Test aggregation
        aggregated = server._default_aggregate(client_models, client_weights)
        
        assert aggregated is not None
        assert 'fc.weight' in aggregated
        assert 'fc.bias' in aggregated
    
    def test_default_aggregation_logic_lora(self):
        """Test default aggregation logic for LoRA mode"""
        lora_config = self.basic_config.copy()
        lora_config['lora_cfg'] = {
            'r': 8,
            'alpha': 16,
            'replaced_modules': ['fc']
        }
        
        server = FederatedServer(
            model_constructor=self.model_constructor,
            clients=self.clients,
            path_manager=self.path_manager,
            config=lora_config,
            device="cpu"
        )
        
        # Initialize global model first
        server.initialize_global_model()
        
        # Mock client models with LoRA parameters
        client_models = [
            {'fc.weight': torch.randn(2, 10), 'fc.bias': torch.randn(2)}
            for _ in range(3)
        ]
        client_weights = [100, 100, 100]
        
        # Test aggregation
        aggregated = server._default_aggregate(client_models, client_weights)
        
        assert aggregated is not None
        assert 'fc.weight' in aggregated
        assert 'fc.bias' in aggregated
    
    def test_default_aggregation_logic_adalora(self):
        """Test default aggregation logic for AdaLoRA mode"""
        adalora_config = self.basic_config.copy()
        adalora_config['adalora_cfg'] = {
            'initial_r': 8,
            'alpha': 16,
            'replaced_modules': ['fc']
        }
        
        server = FederatedServer(
            model_constructor=self.model_constructor,
            clients=self.clients,
            path_manager=self.path_manager,
            config=adalora_config,
            device="cpu"
        )
        
        # Mock client models with AdaLoRA parameters
        client_models = [
            {'fc.weight': torch.randn(2, 10), 'fc.bias': torch.randn(2)}
            for _ in range(3)
        ]
        client_weights = [100, 100, 100]
        
        # Test aggregation
        aggregated = server._default_aggregate(client_models, client_weights)
        
        assert aggregated is not None
        assert 'fc.weight' in aggregated
        assert 'fc.bias' in aggregated
    
    def test_backward_compatibility_original_imports(self):
        """Test that original imports still work"""
        from src.federated import Client, Server, fedavg, lora_fedavg, adalora_fedavg
        from src.federated import FederatedClient, FederatedServer, FederatedAggregator
        
        # Original classes should be available
        assert Server is not None
        assert Client is not None
        assert fedavg is not None
        assert lora_fedavg is not None
        assert adalora_fedavg is not None
        
        # OOP classes should also be available
        assert FederatedServer is not None
        assert FederatedClient is not None
        assert FederatedAggregator is not None
    
    def test_oop_components_independent_functionality(self):
        """Test OOP components can work independently"""
        # Test client independently
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=100)
        self.mock_loader.dataset = mock_dataset
        
        client = FederatedClient(
            client_id=0,
            model_ctor=self.model_constructor,
            train_data_loader=self.mock_loader,
            config={'optimizer': {'name': 'adam', 'lr': 0.001}},
            device="cpu"
        )
        
        # Test aggregator independently
        aggregator = FederatedAggregator()
        
        # Test strategies independently
        strategies = [
            FedAvgStrategy(),
            LoRAFedAvgStrategy(),
            AdaLoRAFedAvgStrategy()
        ]
        
        for strategy in strategies:
            assert strategy.get_name() is not None
            assert strategy.get_description() is not None
    
    def test_error_handling(self):
        """Test error handling in OOP components"""
        # Test server initialization error - should raise ServerConfigurationError
        with pytest.raises(ServerConfigurationError):
            FederatedServer(
                model_constructor=None,  # Invalid constructor
                clients=self.clients,
                path_manager=self.path_manager,
                config=self.basic_config,
                device="cpu"
            )
        
        # Test aggregation error handling
        server = FederatedServer(
            model_constructor=self.model_constructor,
            clients=self.clients,
            path_manager=self.path_manager,
            config=self.basic_config,
            device="cpu"
        )
        
        # Test with invalid client models
        with pytest.raises(ServerOperationError):
            server.aggregate_models([], [])  # Empty models
    
    def test_path_manager_integration(self):
        """Test PathManager integration with OOP components"""
        # Test different output paths for different modes
        standard_pm = PathManager(
            root=self.temp_dir,
            dataset_name="test",
            model_name="simple",
            use_lora=False,
            use_adalora=False
        )
        
        lora_pm = PathManager(
            root=self.temp_dir,
            dataset_name="test",
            model_name="simple",
            use_lora=True,
            use_adalora=False
        )
        
        adalora_pm = PathManager(
            root=self.temp_dir,
            dataset_name="test",
            model_name="simple",
            use_lora=False,
            use_adalora=True
        )
        
        # Verify different paths are generated
        assert standard_pm.run_root != lora_pm.run_root
        assert lora_pm.run_root != adalora_pm.run_root
        assert standard_pm.run_root != adalora_pm.run_root
    
    def teardown_method(self):
        """Clean up test environment"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)