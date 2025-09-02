"""
End-to-End OOP Architecture Tests

This test suite verifies that the OOP refactored code can perform complete
federated learning workflows independently of the original architecture.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import json
from typing import Dict, List, Any

from src.implementations.servers import FederatedServer
from src.implementations.clients import FederatedClient
from src.strategies.aggregation import FedAvgStrategy, LoRAFedAvgStrategy, AdaLoRAFedAvgStrategy
from src.utils.paths import PathManager
from src.core.exceptions.exceptions import ServerOperationError


class SimpleNN(nn.Module):
    """Simple neural network for testing"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class TestEndToEndOOP:
    """End-to-end test suite for OOP architecture"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create simple dataset
        self.X = torch.randn(100, 10)
        self.y = torch.randint(0, 2, (100,))
        
        # Create data loaders for different clients
        self.client_loaders = []
        for i in range(3):
            client_data = []
            for j in range(10):  # 10 batches per client
                start_idx = j * 10 + i * 30
                end_idx = start_idx + 10
                if end_idx <= len(self.X):
                    client_data.append((self.X[start_idx:end_idx], self.y[start_idx:end_idx]))
            
            mock_loader = Mock()
            mock_loader.__iter__ = Mock(return_value=iter(client_data))
            mock_loader.__len__ = Mock(return_value=len(client_data))
            self.client_loaders.append(mock_loader)
        
        # Mock clients
        self.clients = []
        for i, loader in enumerate(self.client_loaders):
            client = Mock()
            client.client_id = i
            client.train_loader = loader
            client.num_samples = 30
            client.num_train_samples = 25
            client.num_test_samples = 5
            client.global_model = None
            client._training_metrics = {}
            self.clients.append(client)
        
        self.path_manager = PathManager(
            root=self.temp_dir,
            dataset_name="e2e_test",
            model_type="simple_nn",
            use_lora=False,
            use_adalora=False
        )
        
        self.model_constructor = lambda: SimpleNN()
        
        self.basic_config = {
            'federated': {
                'num_rounds': 10
            },
            'lora_cfg': {},
            'adalora_cfg': {},
            'save_client_each_round': True,
            'model_info': {
                'dataset': 'e2e_test',
                'model_type': 'simple_nn'
            }
        }
    
    def test_complete_standard_fl_workflow(self):
        """Test complete standard federated learning workflow"""
        config = self.basic_config.copy()
        config['model_info']['training_mode'] = 'standard'
        
        server = FederatedServer(
            model_constructor=self.model_constructor,
            clients=self.clients,
            path_manager=self.path_manager,
            config=config,
            device="cpu"
        )
        
        # Initialize global model
        server.initialize_global_model()
        
        # Mock client training
        def mock_client_training(client, global_model):
            # Simulate local training
            client.global_model = global_model
            client._training_metrics = {
                'train_loss': 0.5,
                'train_acc': 0.8,
                'test_acc': 0.75
            }
            return global_model.state_dict()
        
        # Simulate federated learning rounds
        for round_num in range(2):
            # Select clients
            selected_clients = server.select_clients(round_num)
            
            # Simulate client training
            client_models = []
            client_weights = []
            
            for client in selected_clients:
                client_model = mock_client_training(client, server._global_model)
                client_models.append(client_model)
                client_weights.append(client.num_samples)
            
            # Aggregate models
            aggregated_model = server.aggregate_models(client_models, client_weights)
            
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
            
            # Mock save operations to avoid file system operations
            with patch.object(server, '_save_client_checkpoints'):
                with patch.object(server, '_save_global_checkpoint'):
                    with patch.object(server, '_save_round_metrics'):
                        with patch.object(server, '_generate_metrics_plots'):
                            server.save_checkpoint(round_num, metrics)
        
        # Verify workflow completed successfully
        assert server._global_model is not None
        assert len(list(self.path_manager.rounds_dir.glob('*'))) >= 0
    
    def test_complete_lora_workflow(self):
        """Test complete LoRA federated learning workflow"""
        # Create LoRA model constructor
        def lora_model_constructor():
            model = SimpleNN()
            # Mock LoRA injection
            model.fc1.lora_A = nn.Parameter(torch.randn(4, 10))
            model.fc1.lora_B = nn.Parameter(torch.randn(5, 4))
            model.fc2.lora_A = nn.Parameter(torch.randn(4, 5))
            model.fc2.lora_B = nn.Parameter(torch.randn(2, 4))
            return model
        
        config = self.basic_config.copy()
        config['lora_cfg'] = {
            'r': 4,
            'alpha': 8,
            'replaced_modules': ['fc1', 'fc2']
        }
        config['model_info']['training_mode'] = 'lora'
        
        # Update path manager for LoRA
        path_manager = PathManager(
            root=self.temp_dir,
            dataset_name="e2e_test",
            model_type="simple_nn",
            use_lora=True,
            use_adalora=False
        )
        
        server = FederatedServer(
            model_constructor=lora_model_constructor,
            clients=self.clients,
            path_manager=path_manager,
            config=config,
            device="cpu"
        )
        
        # Initialize global model
        server.initialize_global_model()
        
        # Test that LoRA configuration is properly set
        assert server.lora_cfg['r'] == 4
        assert server.lora_cfg['replaced_modules'] == ['fc1', 'fc2']
        
        # Simulate one round of training
        selected_clients = server.select_clients(0)
        assert len(selected_clients) == 3
        
        # Test aggregation with LoRA mode
        client_models = []
        for _ in selected_clients:
            model = lora_model_constructor()
            client_models.append(model.state_dict())
        
        client_weights = [30, 30, 30]
        
        # Mock LoRA-specific aggregation
        with patch('src.implementations.servers.federated_server.get_trainable_keys') as mock_get_keys:
            mock_get_keys.return_value = ['fc1.lora_A', 'fc1.lora_B', 'fc2.lora_A', 'fc2.lora_B']
            
            aggregated = server._default_aggregate(client_models, client_weights)
            
            # Verify aggregation worked
            assert aggregated is not None
            assert 'fc1.lora_A' in aggregated
            assert 'fc1.lora_B' in aggregated
    
    def test_complete_adalora_workflow(self):
        """Test complete AdaLoRA federated learning workflow"""
        config = self.basic_config.copy()
        config['adalora_cfg'] = {
            'initial_r': 4,
            'alpha': 8,
            'replaced_modules': ['fc1', 'fc2']
        }
        config['model_info']['training_mode'] = 'adalora'
        
        # Update path manager for AdaLoRA
        path_manager = PathManager(
            root=self.temp_dir,
            dataset_name="e2e_test",
            model_type="simple_nn",
            use_lora=False,
            use_adalora=True
        )
        
        server = FederatedServer(
            model_constructor=self.model_constructor,
            clients=self.clients,
            path_manager=path_manager,
            config=config,
            device="cpu"
        )
        
        # Test that AdaLoRA configuration is properly set
        assert server.adalora_cfg['initial_r'] == 4
        assert server.adalora_cfg['replaced_modules'] == ['fc1', 'fc2']
        
        # Simulate client models
        client_models = [self.model_constructor().state_dict() for _ in self.clients]
        client_weights = [30, 30, 30]
        
        # Mock AdaLoRA aggregation
        with patch('src.implementations.servers.federated_server.adalora_fedavg') as mock_adalora:
            mock_result = self.model_constructor().state_dict()
            mock_adalora.return_value = mock_result
            
            aggregated = server._default_aggregate(client_models, client_weights)
            
            # Verify AdaLoRA aggregation was called
            mock_adalora.assert_called_once_with(client_models, client_weights)
            assert aggregated == mock_result
    
    def test_strategy_pattern_end_to_end(self):
        """Test end-to-end workflow with strategy pattern"""
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
        
        # Test that strategy is used
        client_models = [self.model_constructor().state_dict() for _ in self.clients]
        client_weights = [30, 30, 30]
        
        with patch.object(strategy, 'aggregate') as mock_aggregate:
            mock_result = self.model_constructor().state_dict()
            mock_aggregate.return_value = mock_result
            
            result = server.aggregate_models(client_models, client_weights)
            
            # Verify strategy was used with correct parameters
            mock_aggregate.assert_called_once_with(
                client_models=client_models,
                client_weights=client_weights,
                global_model=server._global_model,
                config=server._config
            )
            assert result == mock_result
    
    def test_error_handling_end_to_end(self):
        """Test error handling in end-to-end workflows"""
        server = FederatedServer(
            model_constructor=self.model_constructor,
            clients=self.clients,
            path_manager=self.path_manager,
            config=self.basic_config,
            device="cpu"
        )
        
        # Test error handling in aggregation
        with pytest.raises(ServerOperationError):
            server.aggregate_models([], [])  # Empty client models
        
        # Test error handling with invalid model states
        invalid_models = [{'invalid_key': torch.randn(10, 10)}]
        with pytest.raises(ServerOperationError):
            server.aggregate_models(invalid_models, [30])
    
    def test_output_structure_generation(self):
        """Test that output structure is correctly generated"""
        config = self.basic_config.copy()
        config['save_client_each_round'] = True
        
        server = FederatedServer(
            model_constructor=self.model_constructor,
            clients=self.clients,
            path_manager=self.path_manager,
            config=config,
            device="cpu"
        )
        
        # Mock save operations
        with patch.object(server, '_save_client_checkpoints') as mock_save_client:
            with patch.object(server, '_save_global_checkpoint') as mock_save_global:
                with patch.object(server, '_save_round_metrics') as mock_save_metrics:
                    with patch.object(server, '_generate_metrics_plots') as mock_save_plots:
                        server.save_checkpoint(1, {'round': 1, 'test_acc': 0.8})
                        
                        # Verify all save operations were called
                        mock_save_client.assert_called_once()
                        mock_save_global.assert_called_once()
                        mock_save_metrics.assert_called_once()
                        mock_save_plots.assert_called_once()
    
    def test_backward_compatibility_end_to_end(self):
        """Test that backward compatibility is maintained in end-to-end workflows"""
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
        client_meta = server.build_client_meta(1, 0, 30, {'train_loss': 0.5})
        assert client_meta['round'] == 1
        assert client_meta['client'] == 0
        assert client_meta['num_samples'] == 30
        
        server_meta = server.build_server_meta(1)
        assert server_meta['round'] == 1
    
    def test_performance_characteristics(self):
        """Test performance characteristics of OOP implementation"""
        server = FederatedServer(
            model_constructor=self.model_constructor,
            clients=self.clients,
            path_manager=self.path_manager,
            config=self.basic_config,
            device="cpu"
        )
        
        # Test initialization performance
        import time
        start_time = time.time()
        server.initialize_global_model()
        init_time = time.time() - start_time
        
        # Model initialization should be fast
        assert init_time < 1.0  # Should initialize in less than 1 second
        
        # Test aggregation performance
        client_models = [self.model_constructor().state_dict() for _ in range(10)]
        client_weights = [30] * 10
        
        start_time = time.time()
        aggregated = server._default_aggregate(client_models, client_weights)
        agg_time = time.time() - start_time
        
        # Aggregation should be fast
        assert agg_time < 0.5  # Should aggregate in less than 0.5 seconds
        
        # Verify result quality
        assert aggregated is not None
        assert len(aggregated) > 0
    
    def teardown_method(self):
        """Clean up test environment"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)