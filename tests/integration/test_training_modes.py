"""
Comprehensive Training Mode Tests

This test suite verifies all three training modes (Standard, LoRA, AdaLoRA)
work correctly with the OOP refactored code.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
from typing import Dict, List, Any

from src.implementations.servers import FederatedServer
from src.implementations.clients import FederatedClient
from src.strategies.aggregation import FedAvgStrategy, LoRAFedAvgStrategy, AdaLoRAFedAvgStrategy
from src.utils.paths import PathManager
from src.strategies.training.lora_manager import LoRAManager
from src.strategies.training.adalora_manager import AdaLoRAManager


class TestModel(nn.Module):
    """Test model with multiple layers for LoRA/AdaLoRA testing"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)
        self.fc3 = nn.Linear(5, 2)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class TestTrainingModes:
    """Test suite for different training modes"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.model_constructor = lambda: TestModel()
        
        # Mock data loader
        self.mock_loader = Mock()
        self.mock_loader.__iter__ = Mock(return_value=iter([
            (torch.randn(16, 10), torch.randint(0, 2, (16,)))
            for _ in range(3)
        ]))
        
        # Mock clients
        self.clients = [
            Mock(client_id=i, train_loader=self.mock_loader, num_samples=50)
            for i in range(2)
        ]
        
        self.path_manager = PathManager(
            root=self.temp_dir,
            dataset_name="test",
            model_type="test_model",
            use_lora=False,
            use_adalora=False
        )
    
    def test_standard_federated_learning(self):
        """Test standard federated learning mode"""
        config = {
            'federated': {
                'num_rounds': 10
            },
            'lora_cfg': {},
            'adalora_cfg': {},
            'save_client_each_round': False,
            'model_info': {'dataset': 'test', 'model_type': 'standard'}
        }
        
        server = FederatedServer(
            model_constructor=self.model_constructor,
            clients=self.clients,
            path_manager=self.path_manager,
            config=config,
            device="cpu"
        )
        
        # Initialize global model
        server.initialize_global_model()
        
        # Test client selection
        selected_clients = server.select_clients(1)
        assert len(selected_clients) == 2
        
        # Test aggregation with standard mode
        client_models = [
            {'fc1.weight': torch.randn(20, 10), 'fc1.bias': torch.randn(20),
             'fc2.weight': torch.randn(5, 20), 'fc2.bias': torch.randn(5),
             'fc3.weight': torch.randn(2, 5), 'fc3.bias': torch.randn(2)}
            for _ in range(2)
        ]
        client_weights = [50, 50]
        
        aggregated = server._default_aggregate(client_models, client_weights)
        
        # Verify all parameters are aggregated
        expected_keys = {'fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias', 'fc3.weight', 'fc3.bias'}
        assert set(aggregated.keys()) == expected_keys
        
        # Verify aggregation is weighted average
        for key in expected_keys:
            expected = (client_models[0][key] + client_models[1][key]) / 2
            assert torch.allclose(aggregated[key], expected, atol=1e-6)
    
    def test_lora_federated_learning(self):
        """Test LoRA federated learning mode"""
        # Create model with LoRA
        model = TestModel()
        lora_manager = LoRAManager(r=8, alpha=16, target_modules=['Linear'])
        replaced_modules = lora_manager.inject_lora_modules(model)
        lora_manager.mark_only_lora_as_trainable(model, train_classifier_head=True)
        
        # Update model constructor to return LoRA model
        def lora_model_constructor():
            model = TestModel()
            lora_manager = LoRAManager(r=8, alpha=16, target_modules=['Linear'])
            lora_manager.inject_lora_modules(model)
            lora_manager.mark_only_lora_as_trainable(model, train_classifier_head=True)
            return model
        
        config = {
            'federated': {
                'num_rounds': 10
            },
            'lora_cfg': {
                'r': 8,
                'alpha': 16,
                'replaced_modules': replaced_modules
            },
            'adalora_cfg': {},
            'save_client_each_round': False,
            'model_info': {'dataset': 'test', 'model_type': 'lora'}
        }
        
        server = FederatedServer(
            model_constructor=lora_model_constructor,
            clients=self.clients,
            path_manager=self.path_manager,
            config=config,
            device="cpu"
        )
        
        # Initialize global model
        server.initialize_global_model()
        
        # Test aggregation with LoRA mode
        client_models = [
            {
                'fc1.weight': torch.randn(20, 10), 'fc1.bias': torch.randn(20),
                'fc1.lora_A': torch.randn(8, 10), 'fc1.lora_B': torch.randn(20, 8),
                'fc2.weight': torch.randn(5, 20), 'fc2.bias': torch.randn(5),
                'fc2.lora_A': torch.randn(8, 20), 'fc2.lora_B': torch.randn(5, 8),
                'fc3.weight': torch.randn(2, 5), 'fc3.bias': torch.randn(2),
                'fc3.lora_A': torch.randn(8, 5), 'fc3.lora_B': torch.randn(2, 8)
            }
            for _ in range(2)
        ]
        client_weights = [50, 50]
        
        aggregated = server._default_aggregate(client_models, client_weights)
        
        # Verify LoRA parameters are aggregated
        assert 'fc1.lora_A' in aggregated
        assert 'fc1.lora_B' in aggregated
        assert 'fc2.lora_A' in aggregated
        assert 'fc2.lora_B' in aggregated
        assert 'fc3.lora_A' in aggregated
        assert 'fc3.lora_B' in aggregated
    
    def test_adalora_federated_learning(self):
        """Test AdaLoRA federated learning mode"""
        # Create model with AdaLoRA
        model = TestModel()
        adalora_manager = AdaLoRAManager(r=8, alpha=16, target_modules=['Linear'])
        replaced_modules = adalora_manager.inject_adalora_modules(model)
        adalora_manager.mark_only_adalora_as_trainable(model, train_classifier_head=True)
        
        # Update model constructor to return AdaLoRA model
        def adalora_model_constructor():
            model = TestModel()
            adalora_manager = AdaLoRAManager(r=8, alpha=16, target_modules=['Linear'])
            adalora_manager.inject_adalora_modules(model)
            adalora_manager.mark_only_adalora_as_trainable(model, train_classifier_head=True)
            return model
        
        config = {
            'federated': {
                'num_rounds': 10
            },
            'lora_cfg': {},
            'adalora_cfg': {
                'initial_r': 8,
                'alpha': 16,
                'replaced_modules': replaced_modules
            },
            'save_client_each_round': False,
            'model_info': {'dataset': 'test', 'model_type': 'adalora'}
        }
        
        server = FederatedServer(
            model_constructor=adalora_model_constructor,
            clients=self.clients,
            path_manager=self.path_manager,
            config=config,
            device="cpu"
        )
        
        # Initialize global model
        server.initialize_global_model()
        
        # Test aggregation with AdaLoRA mode
        client_models = [
            {
                'fc1.weight': torch.randn(20, 10), 'fc1.bias': torch.randn(20),
                'fc2.weight': torch.randn(5, 20), 'fc2.bias': torch.randn(5),
                'fc3.weight': torch.randn(2, 5), 'fc3.bias': torch.randn(2)
            }
            for _ in range(2)
        ]
        client_weights = [50, 50]
        
        # Mock the adalora_fedavg function
        with patch('src.implementations.servers.federated_server.adalora_fedavg') as mock_adalora_fedavg:
            mock_result = {
                'fc1.weight': torch.randn(20, 10), 'fc1.bias': torch.randn(20),
                'fc2.weight': torch.randn(5, 20), 'fc2.bias': torch.randn(5),
                'fc3.weight': torch.randn(2, 5), 'fc3.bias': torch.randn(2)
            }
            mock_adalora_fedavg.return_value = mock_result
            
            aggregated = server._default_aggregate(client_models, client_weights)
            
            # Verify adalora_fedavg was called
            mock_adalora_fedavg.assert_called_once_with(client_models, client_weights)
            
            # Verify result is returned
            assert aggregated == mock_result
    
    def test_mode_detection_priority(self):
        """Test that AdaLoRA has priority over LoRA, which has priority over Standard"""
        # Test AdaLoRA mode (highest priority)
        config = {
            'federated': {
                'num_rounds': 10
            },
            'lora_cfg': {'replaced_modules': ['fc1']},
            'adalora_cfg': {'replaced_modules': ['fc1']},
            'save_client_each_round': False,
            'model_info': {'dataset': 'test', 'model_type': 'adalora'}
        }
        
        server = FederatedServer(
            model_constructor=self.model_constructor,
            clients=self.clients,
            path_manager=self.path_manager,
            config=config,
            device="cpu"
        )
        
        client_models = [{'fc1.weight': torch.randn(20, 10)}]
        client_weights = [50]
        
        with patch('src.implementations.servers.federated_server.adalora_fedavg') as mock_adalora:
            with patch('src.implementations.servers.federated_server.lora_fedavg') as mock_lora:
                with patch('src.implementations.servers.federated_server.fedavg') as mock_fedavg:
                    server._default_aggregate(client_models, client_weights)
                    
                    # Only adalora_fedavg should be called
                    mock_adalora.assert_called_once()
                    mock_lora.assert_not_called()
                    mock_fedavg.assert_not_called()
    
    def test_strategy_pattern_integration(self):
        """Test strategy pattern integration with different training modes"""
        # Test with FedAvg strategy
        fedavg_strategy = FedAvgStrategy()
        config = {
            'federated': {
                'num_rounds': 10
            },
            'lora_cfg': {},
            'adalora_cfg': {},
            'save_client_each_round': False,
            'model_info': {'dataset': 'test', 'model_type': 'strategy_test'}
        }
        
        server = FederatedServer(
            model_constructor=self.model_constructor,
            clients=self.clients,
            path_manager=self.path_manager,
            config=config,
            device="cpu",
            aggregation_strategy=fedavg_strategy
        )
        
        # Test aggregation uses strategy
        client_models = [{'fc1.weight': torch.randn(20, 10)}]
        client_weights = [50]
        
        with patch.object(fedavg_strategy, 'aggregate') as mock_aggregate:
            mock_result = {'fc1.weight': torch.randn(20, 10)}
            mock_aggregate.return_value = mock_result
            
            result = server.aggregate_models(client_models, client_weights)
            
            # Verify strategy was used
            mock_aggregate.assert_called_once()
            assert result == mock_result
    
    def test_path_generation_for_different_modes(self):
        """Test that different training modes generate correct output paths"""
        # Standard mode
        standard_pm = PathManager(
            root=self.temp_dir,
            dataset_name="test",
            model_type="test",
            use_lora=False,
            use_adalora=False
        )
        
        # LoRA mode
        lora_pm = PathManager(
            root=self.temp_dir,
            dataset_name="test",
            model_type="test",
            use_lora=True,
            use_adalora=False
        )
        
        # AdaLoRA mode
        adalora_pm = PathManager(
            root=self.temp_dir,
            dataset_name="test",
            model_type="test",
            use_lora=False,
            use_adalora=True
        )
        
        # Verify paths are different and correctly formatted
        assert 'models' in standard_pm.root_dir
        assert 'loras' in lora_pm.root_dir
        assert 'adaloras' in adalora_pm.root_dir
        
        # Verify no mode conflicts
        assert standard_pm.root_dir != lora_pm.root_dir
        assert lora_pm.root_dir != adalora_pm.root_dir
        assert standard_pm.root_dir != adalora_pm.root_dir
    
    def test_backward_compatibility_training_methods(self):
        """Test that training methods maintain backward compatibility"""
        config = {
            'federated': {
                'num_rounds': 10
            },
            'lora_cfg': {},
            'adalora_cfg': {},
            'save_client_each_round': False,
            'model_info': {'dataset': 'test', 'model_type': 'backward_compatibility'}
        }
        
        server = FederatedServer(
            model_constructor=self.model_constructor,
            clients=self.clients,
            path_manager=self.path_manager,
            config=config,
            device="cpu"
        )
        
        # Test backward compatibility methods exist
        assert hasattr(server, 'run')
        assert hasattr(server, 'model_ctor')
        assert hasattr(server, 'clients')
        assert hasattr(server, 'paths')
        assert hasattr(server, 'lora_cfg')
        assert hasattr(server, 'adalora_cfg')
        
        # Test method functionality
        assert server.model_ctor() is not None
        assert len(server.clients) == 2
        assert isinstance(server.paths, PathManager)
        assert isinstance(server.lora_cfg, dict)
        assert isinstance(server.adalora_cfg, dict)
    
    def teardown_method(self):
        """Clean up test environment"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)