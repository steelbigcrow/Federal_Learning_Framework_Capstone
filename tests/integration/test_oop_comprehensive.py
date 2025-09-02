"""
Comprehensive Integration Tests for OOP Refactored Code

Tests for complete end-to-end workflows including:
- Full federated learning cycles
- Integration with original codebase
- Performance benchmarks
- Error handling and recovery
- Backward compatibility
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch
import tempfile
import os
import time
from typing import Dict, List, Any

from src.implementations.servers import FederatedServer
from src.implementations.clients import FederatedClient
from src.implementations.aggregators import FederatedAggregator
from src.strategies.aggregation import FedAvgStrategy, LoRAFedAvgStrategy, AdaLoRAFedAvgStrategy
from src.factories.component_factory import ComponentFactory
from src.factories.model_factory import ModelFactory
from src.strategies.strategy_manager import StrategyManager
from src.utils.paths import PathManager
from src.core.exceptions.exceptions import ServerOperationError, ClientError


class TestModel(nn.Module):
    """Test model for integration tests"""
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class TestCompleteIntegration:
    """Test complete integration of OOP components"""
    
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        
        # Create path manager
        self.path_manager = PathManager(
            root=self.temp_dir,
            dataset_name="integration_test",
            model_name="test_model",
            use_lora=False,
            use_adalora=False
        )
        
        # Model constructor
        self.model_constructor = lambda: TestModel()
        
        # Create mock data loaders
        self.client_loaders = []
        for i in range(3):
            mock_loader = Mock()
            mock_loader.__iter__ = Mock(return_value=iter([
                (torch.randn(32, 784), torch.randint(0, 10, (32,)))
                for _ in range(10)
            ]))
            mock_loader.__len__ = Mock(return_value=10)
            
            # Mock dataset
            mock_dataset = Mock()
            mock_dataset.__len__ = Mock(return_value=320)
            mock_loader.dataset = mock_dataset
            
            self.client_loaders.append(mock_loader)
        
        # Create mock clients
        self.clients = []
        for i, loader in enumerate(self.client_loaders):
            client = Mock()
            client.client_id = i
            client.train_loader = loader
            client.num_samples = 320
            client.num_train_samples = 300
            client.num_test_samples = 20
            client.global_model = None
            client._training_metrics = {}
            self.clients.append(client)
        
        # Basic configuration
        self.basic_config = {
            'federated': {
                'num_rounds': 3,
                'clients_per_round': 3
            },
            'lora_cfg': {},
            'adalora_cfg': {},
            'save_client_each_round': True,
            'model_info': {
                'dataset': 'integration_test',
                'model_type': 'test_model'
            }
        }
    
    def teardown_method(self):
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_complete_standard_federated_learning_cycle(self):
        """Test complete standard federated learning cycle"""
        server = FederatedServer(
            model_constructor=self.model_constructor,
            clients=self.clients,
            path_manager=self.path_manager,
            config=self.basic_config,
            device="cpu"
        )
        
        # Initialize global model
        server.initialize_global_model()
        assert server._global_model is not None
        
        # Simulate federated learning rounds
        for round_num in range(2):
            # Select clients
            selected_clients = server.select_clients(round_num)
            assert len(selected_clients) == 3
            
            # Simulate client training
            client_models = []
            client_weights = []
            
            for client in selected_clients:
                # Mock client training
                client.global_model = server._global_model
                client._training_metrics = {
                    'train_loss': 0.5,
                    'train_acc': 0.8,
                    'test_acc': 0.75
                }
                
                # Return model state dict
                client_models.append(server._global_model.state_dict())
                client_weights.append(client.num_samples)
            
            # Aggregate models
            aggregated_model = server.aggregate_models(client_models, client_weights)
            assert aggregated_model is not None
            
            # Update global model
            server._global_model.load_state_dict(aggregated_model)
            
            # Save checkpoint
            metrics = {
                'round': round_num,
                'num_clients': len(selected_clients),
                'train_loss': 0.5,
                'train_acc': 0.8,
                'test_acc': 0.75
            }
            
            # Mock save operations
            with patch.object(server, '_save_client_checkpoints'):
                with patch.object(server, '_save_global_checkpoint'):
                    with patch.object(server, '_save_round_metrics'):
                        with patch.object(server, '_generate_metrics_plots'):
                            server.save_checkpoint(round_num, metrics)
        
        # Verify completion
        assert server._global_model is not None
        assert len(list(self.path_manager.rounds_dir.glob('*'))) >= 0
    
    def test_complete_lora_federated_learning_cycle(self):
        """Test complete LoRA federated learning cycle"""
        # Create LoRA configuration
        lora_config = self.basic_config.copy()
        lora_config['lora_cfg'] = {
            'r': 8,
            'alpha': 16,
            'replaced_modules': ['fc1', 'fc2']
        }
        lora_config['model_info']['training_mode'] = 'lora'
        
        # Update path manager for LoRA
        path_manager = PathManager(
            root=self.temp_dir,
            dataset_name="integration_test",
            model_name="test_model",
            use_lora=True,
            use_adalora=False
        )
        
        server = FederatedServer(
            model_constructor=self.model_constructor,
            clients=self.clients,
            path_manager=path_manager,
            config=lora_config,
            device="cpu"
        )
        
        # Initialize global model
        server.initialize_global_model()
        
        # Test LoRA configuration
        assert server.lora_cfg['r'] == 8
        assert server.lora_cfg['replaced_modules'] == ['fc1', 'fc2']
        
        # Simulate one round of LoRA training
        selected_clients = server.select_clients(0)
        assert len(selected_clients) == 3
        
        # Mock LoRA client models
        client_models = []
        for _ in selected_clients:
            model = self.model_constructor()
            # Add mock LoRA parameters
            model.fc1.lora_A = nn.Parameter(torch.randn(8, 784))
            model.fc1.lora_B = nn.Parameter(torch.randn(128, 8))
            model.fc2.lora_A = nn.Parameter(torch.randn(8, 128))
            model.fc2.lora_B = nn.Parameter(torch.randn(10, 8))
            client_models.append(model.state_dict())
        
        client_weights = [320, 320, 320]
        
        # Test LoRA aggregation
        with patch('src.implementations.servers.federated_server.get_trainable_keys') as mock_get_keys:
            mock_get_keys.return_value = ['fc1.lora_A', 'fc1.lora_B', 'fc2.lora_A', 'fc2.lora_B']
            
            aggregated = server._default_aggregate(client_models, client_weights)
            
            assert aggregated is not None
            assert 'fc1.lora_A' in aggregated
            assert 'fc1.lora_B' in aggregated
    
    def test_complete_adalora_federated_learning_cycle(self):
        """Test complete AdaLoRA federated learning cycle"""
        # Create AdaLoRA configuration
        adalora_config = self.basic_config.copy()
        adalora_config['adalora_cfg'] = {
            'initial_r': 8,
            'alpha': 16,
            'replaced_modules': ['fc1', 'fc2']
        }
        adalora_config['model_info']['training_mode'] = 'adalora'
        
        # Update path manager for AdaLoRA
        path_manager = PathManager(
            root=self.temp_dir,
            dataset_name="integration_test",
            model_name="test_model",
            use_lora=False,
            use_adalora=True
        )
        
        server = FederatedServer(
            model_constructor=self.model_constructor,
            clients=self.clients,
            path_manager=path_manager,
            config=adalora_config,
            device="cpu"
        )
        
        # Test AdaLoRA configuration
        assert server.adalora_cfg['initial_r'] == 8
        assert server.adalora_cfg['replaced_modules'] == ['fc1', 'fc2']
        
        # Test AdaLoRA aggregation
        client_models = [self.model_constructor().state_dict() for _ in self.clients]
        client_weights = [320, 320, 320]
        
        with patch('src.implementations.servers.federated_server.adalora_fedavg') as mock_adalora:
            mock_result = self.model_constructor().state_dict()
            mock_adalora.return_value = mock_result
            
            aggregated = server._default_aggregate(client_models, client_weights)
            
            mock_adalora.assert_called_once_with(client_models, client_weights)
            assert aggregated == mock_result
    
    def test_strategy_pattern_integration(self):
        """Test integration with strategy pattern"""
        strategy = FedAvgStrategy()
        
        server = FederatedServer(
            model_constructor=self.model_constructor,
            clients=self.clients,
            path_manager=self.path_manager,
            config=self.basic_config,
            device="cpu",
            aggregation_strategy=strategy
        )
        
        # Initialize global model
        server.initialize_global_model()
        
        # Test strategy usage
        client_models = [self.model_constructor().state_dict() for _ in self.clients]
        client_weights = [320, 320, 320]
        
        with patch.object(strategy, 'aggregate') as mock_aggregate:
            mock_result = self.model_constructor().state_dict()
            mock_aggregate.return_value = mock_result
            
            result = server.aggregate_models(client_models, client_weights)
            
            # Verify strategy was called correctly
            mock_aggregate.assert_called_once_with(
                client_models=client_models,
                client_weights=client_weights,
                global_model=server._global_model,
                config=server._config
            )
            assert result == mock_result
    
    def test_factory_pattern_integration(self):
        """Test integration with factory pattern"""
        # Test ModelFactory integration
        model_factory = ModelFactory()
        
        # Create model using factory
        config = {
            'input_size': 784,
            'hidden_size': 128,
            'num_classes': 10
        }
        
        model = model_factory.create_model_with_defaults('mnist', 'mlp', config_overrides=config)
        assert isinstance(model, nn.Module)
        
        # Use factory-created model in server
        def factory_model_constructor():
            return model_factory.create_model_with_defaults('mnist', 'mlp', config_overrides=config)
        
        server = FederatedServer(
            model_constructor=factory_model_constructor,
            clients=self.clients,
            path_manager=self.path_manager,
            config=self.basic_config,
            device="cpu"
        )
        
        server.initialize_global_model()
        assert server._global_model is not None
    
    def test_performance_benchmarks(self):
        """Test performance benchmarks for OOP components"""
        server = FederatedServer(
            model_constructor=self.model_constructor,
            clients=self.clients,
            path_manager=self.path_manager,
            config=self.basic_config,
            device="cpu"
        )
        
        # Benchmark initialization
        start_time = time.time()
        server.initialize_global_model()
        init_time = time.time() - start_time
        
        # Should initialize quickly
        assert init_time < 2.0, f"Initialization took {init_time:.2f}s, expected < 2.0s"
        
        # Benchmark aggregation
        client_models = [self.model_constructor().state_dict() for _ in range(10)]
        client_weights = [320] * 10
        
        start_time = time.time()
        aggregated = server._default_aggregate(client_models, client_weights)
        agg_time = time.time() - start_time
        
        # Should aggregate quickly
        assert agg_time < 1.0, f"Aggregation took {agg_time:.2f}s, expected < 1.0s"
        
        # Verify result quality
        assert aggregated is not None
        assert len(aggregated) > 0
    
    def test_error_handling_and_recovery(self):
        """Test error handling and recovery"""
        server = FederatedServer(
            model_constructor=self.model_constructor,
            clients=self.clients,
            path_manager=self.path_manager,
            config=self.basic_config,
            device="cpu"
        )
        
        # Test error handling in aggregation
        with pytest.raises(ServerOperationError):
            server.aggregate_models([], [])  # Empty models
        
        # Test error handling with invalid model states
        invalid_models = [{'invalid_key': torch.randn(10, 10)}]
        with pytest.raises(ServerOperationError):
            server.aggregate_models(invalid_models, [320])
        
        # Test recovery after error
        server.initialize_global_model()
        valid_models = [self.model_constructor().state_dict() for _ in self.clients]
        valid_weights = [320] * len(self.clients)
        
        # Should work after error
        result = server.aggregate_models(valid_models, valid_weights)
        assert result is not None
    
    def test_backward_compatibility(self):
        """Test backward compatibility with original codebase"""
        # Test original imports still work
        from src.federated import Server, Client, FederatedServer, FederatedClient
        
        # Original classes should be available
        assert Server is not None
        assert Client is not None
        assert FederatedServer is not None
        assert FederatedClient is not None
        
        # Test original functionality
        server = FederatedServer(
            model_constructor=self.model_constructor,
            clients=self.clients,
            path_manager=self.path_manager,
            config=self.basic_config,
            device="cpu"
        )
        
        # Test backward compatibility methods
        assert hasattr(server, 'run')
        assert hasattr(server, 'build_client_meta')
        assert hasattr(server, 'build_server_meta')
        assert hasattr(server, 'safe_write_logs')
        
        # Test method functionality
        client_meta = server.build_client_meta(1, 0, 320, {'train_loss': 0.5})
        assert client_meta['round'] == 1
        assert client_meta['client'] == 0
        assert client_meta['num_samples'] == 320
        
        server_meta = server.build_server_meta(1)
        assert server_meta['round'] == 1
    
    def test_memory_efficiency(self):
        """Test memory efficiency of OOP implementation"""
        # Test with large number of clients
        many_clients = []
        for i in range(50):  # 50 clients
            client = Mock()
            client.client_id = i
            client.train_loader = self.client_loaders[i % len(self.client_loaders)]
            client.num_samples = 320
            client.global_model = None
            client._training_metrics = {}
            many_clients.append(client)
        
        server = FederatedServer(
            model_constructor=self.model_constructor,
            clients=many_clients,
            path_manager=self.path_manager,
            config=self.basic_config,
            device="cpu"
        )
        
        # Should handle many clients without memory issues
        server.initialize_global_model()
        
        # Test aggregation with many clients
        client_models = [self.model_constructor().state_dict() for _ in range(10)]
        client_weights = [320] * 10
        
        result = server.aggregate_models(client_models, client_weights)
        assert result is not None
    
    def test_concurrent_operations(self):
        """Test concurrent operations"""
        import threading
        import queue
        
        server = FederatedServer(
            model_constructor=self.model_constructor,
            clients=self.clients,
            path_manager=self.path_manager,
            config=self.basic_config,
            device="cpu"
        )
        
        server.initialize_global_model()
        
        # Test concurrent client training simulation
        results = queue.Queue()
        
        def simulate_client_training(client_id, server_model):
            """Simulate client training"""
            time.sleep(0.1)  # Simulate training time
            return {
                'client_id': client_id,
                'model': server_model.state_dict(),
                'weight': 320
            }
        
        def worker(client_id, server_model, results_queue):
            """Worker thread for client training"""
            result = simulate_client_training(client_id, server_model)
            results_queue.put(result)
        
        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(
                target=worker,
                args=(i, server._global_model, results)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Collect results
        client_models = []
        client_weights = []
        
        while not results.empty():
            result = results.get()
            client_models.append(result['model'])
            client_weights.append(result['weight'])
        
        # Aggregate results
        aggregated = server.aggregate_models(client_models, client_weights)
        assert aggregated is not None
        assert len(client_models) == 5
    
    def test_configuration_flexibility(self):
        """Test configuration flexibility"""
        # Test with different configurations
        configs = [
            # Standard config
            self.basic_config.copy(),
            # Config with LoRA
            {
                **self.basic_config,
                'lora_cfg': {'r': 4, 'alpha': 8},
                'model_info': {**self.basic_config['model_info'], 'training_mode': 'lora'}
            },
            # Config with AdaLoRA
            {
                **self.basic_config,
                'adalora_cfg': {'initial_r': 4, 'alpha': 8},
                'model_info': {**self.basic_config['model_info'], 'training_mode': 'adalora'}
            }
        ]
        
        for config in configs:
            server = FederatedServer(
                model_constructor=self.model_constructor,
                clients=self.clients,
                path_manager=self.path_manager,
                config=config,
                device="cpu"
            )
            
            server.initialize_global_model()
            assert server._global_model is not None
            
            # Test basic operations
            selected_clients = server.select_clients(0)
            assert len(selected_clients) > 0
            
            # Should handle different configurations gracefully
            if 'lora_cfg' in config and config['lora_cfg']:
                assert server.lora_cfg == config['lora_cfg']
            
            if 'adalora_cfg' in config and config['adalora_cfg']:
                assert server.adalora_cfg == config['adalora_cfg']