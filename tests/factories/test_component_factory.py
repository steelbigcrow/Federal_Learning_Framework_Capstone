"""
Comprehensive OOP Component Factory Tests

Tests for all OOP component factories including:
- ComponentFactory
- ClientFactory  
- ServerFactory
- DatasetFactory
- FactoryRegistry
"""

import pytest
import torch.nn as nn
from unittest.mock import Mock, patch
import tempfile
import os

from src.factories.component_factory import ComponentFactory
from src.factories.client_factory import ClientFactory
from src.factories.server_factory import ServerFactory
from src.factories.dataset_factory import DatasetFactory
from src.factories.factory_registry import FactoryRegistry
from src.core.exceptions.exceptions import FactoryError
from src.implementations.clients.federated_client import FederatedClient
from src.implementations.servers.federated_server import FederatedServer
from src.utils.paths import PathManager


class SimpleTestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
    
    def forward(self, x):
        return self.fc(x)


class TestComponentFactory:
    """Test ComponentFactory functionality"""
    
    def setup_method(self):
        self.factory = ComponentFactory()
    
    def test_initialization(self):
        """Test factory initialization"""
        assert isinstance(self.factory, ComponentFactory)
        assert hasattr(self.factory, 'create')
        assert hasattr(self.factory, 'register')
        assert hasattr(self.factory, 'unregister')
    
    def test_register_and_create_component(self):
        """Test component registration and creation"""
        # Register a test component
        self.factory.register('test_component', SimpleTestModel)
        
        # Create the component
        config = {}
        component = self.factory.create('test_component', config)
        
        assert isinstance(component, SimpleTestModel)
    
    def test_create_unregistered_component(self):
        """Test creating unregistered component"""
        with pytest.raises(FactoryError):
            self.factory.create('unregistered', {})
    
    def test_unregister_component(self):
        """Test component unregistration"""
        # Register and then unregister
        self.factory.register('temp_component', SimpleTestModel)
        self.factory.unregister('temp_component')
        
        # Should fail to create after unregistration
        with pytest.raises(FactoryError):
            self.factory.create('temp_component', {})
    
    def test_list_registered_types(self):
        """Test listing registered component types"""
        # Register some components
        self.factory.register('comp1', SimpleTestModel)
        self.factory.register('comp2', Mock)
        
        types = self.factory.list_registered_types()
        
        assert 'comp1' in types
        assert 'comp2' in types
    
    def test_is_registered(self):
        """Test checking registration status"""
        self.factory.register('test_comp', SimpleTestModel)
        
        assert self.factory.is_registered('test_comp')
        assert not self.factory.is_registered('nonexistent')


class TestClientFactory:
    """Test ClientFactory functionality"""
    
    def setup_method(self):
        self.factory = ClientFactory()
        self.temp_dir = tempfile.mkdtemp()
        self.path_manager = PathManager(
            root=self.temp_dir,
            dataset_name="test",
            model_name="test",
            use_lora=False
        )
        
        # Mock data loader
        self.mock_loader = Mock()
        self.mock_loader.__iter__ = Mock(return_value=iter([
            (torch.randn(32, 10), torch.randint(0, 2, (32,)))
        ]))
        self.mock_loader.__len__ = Mock(return_value=32)
        
        # Mock dataset
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=100)
        self.mock_loader.dataset = mock_dataset
    
    def teardown_method(self):
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_create_federated_client(self):
        """Test creating FederatedClient"""
        config = {
            'client_id': 0,
            'model_ctor': SimpleTestModel,
            'train_data_loader': self.mock_loader,
            'config': {'optimizer': {'name': 'adam', 'lr': 0.001}},
            'device': 'cpu'
        }
        
        client = self.factory.create('federated_client', config)
        
        assert isinstance(client, FederatedClient)
        assert client.id() == 0
    
    def test_create_client_with_validation(self):
        """Test client creation with validation"""
        # Test missing required parameters
        with pytest.raises(FactoryError):
            self.factory.create('federated_client', {})
    
    def test_client_factory_supported_types(self):
        """Test supported client types"""
        types = self.factory.get_supported_client_types()
        
        assert 'federated_client' in types
        assert isinstance(types, list)
    
    def test_client_factory_config_validation(self):
        """Test client configuration validation"""
        # Invalid client ID
        config = {
            'client_id': 'invalid',  # Should be int
            'model_ctor': SimpleTestModel,
            'train_data_loader': self.mock_loader,
            'config': {},
            'device': 'cpu'
        }
        
        with pytest.raises(FactoryError):
            self.factory.create('federated_client', config)


class TestServerFactory:
    """Test ServerFactory functionality"""
    
    def setup_method(self):
        self.factory = ServerFactory()
        self.temp_dir = tempfile.mkdtemp()
        self.path_manager = PathManager(
            root=self.temp_dir,
            dataset_name="test",
            model_name="test",
            use_lora=False
        )
        
        # Mock clients
        self.clients = [Mock(client_id=i, num_samples=100) for i in range(3)]
        
        self.basic_config = {
            'federated': {'num_rounds': 1},
            'lora_cfg': {},
            'adalora_cfg': {},
            'save_client_each_round': False,
            'model_info': {'dataset': 'test', 'model_type': 'simple'}
        }
    
    def teardown_method(self):
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_create_federated_server(self):
        """Test creating FederatedServer"""
        config = {
            'model_constructor': SimpleTestModel,
            'clients': self.clients,
            'path_manager': self.path_manager,
            'config': self.basic_config,
            'device': 'cpu'
        }
        
        server = self.factory.create('federated_server', config)
        
        assert isinstance(server, FederatedServer)
        assert server.model_info == self.basic_config['model_info']
    
    def test_server_factory_supported_types(self):
        """Test supported server types"""
        types = self.factory.get_supported_server_types()
        
        assert 'federated_server' in types
        assert isinstance(types, list)
    
    def test_server_factory_config_validation(self):
        """Test server configuration validation"""
        # Missing required config
        config = {
            'model_constructor': SimpleTestModel,
            'clients': self.clients,
            'path_manager': self.path_manager,
            'config': {},  # Missing required fields
            'device': 'cpu'
        }
        
        with pytest.raises(FactoryError):
            self.factory.create('federated_server', config)


class TestDatasetFactory:
    """Test DatasetFactory functionality"""
    
    def setup_method(self):
        self.factory = DatasetFactory()
    
    def test_create_mnist_dataset(self):
        """Test creating MNIST dataset"""
        config = {
            'batch_size': 32,
            'num_clients': 10,
            'partition_method': 'label_shift'
        }
        
        with patch('src.datasets.mnist.MNISTDataset') as mock_dataset:
            mock_instance = Mock()
            mock_dataset.return_value = mock_instance
            
            dataset = self.factory.create('mnist', config)
            
            assert dataset == mock_instance
            mock_dataset.assert_called_once_with(**config)
    
    def test_create_imdb_dataset(self):
        """Test creating IMDB dataset"""
        config = {
            'batch_size': 32,
            'num_clients': 10,
            'partition_method': 'balanced'
        }
        
        with patch('src.datasets.imdb.IMDBDataset') as mock_dataset:
            mock_instance = Mock()
            mock_dataset.return_value = mock_instance
            
            dataset = self.factory.create('imdb', config)
            
            assert dataset == mock_instance
            mock_dataset.assert_called_once_with(**config)
    
    def test_create_unsupported_dataset(self):
        """Test creating unsupported dataset"""
        with pytest.raises(FactoryError):
            self.factory.create('unsupported', {})
    
    def test_dataset_factory_supported_types(self):
        """Test supported dataset types"""
        types = self.factory.get_supported_datasets()
        
        assert 'mnist' in types
        assert 'imdb' in types
        assert isinstance(types, list)


class TestFactoryRegistry:
    """Test FactoryRegistry functionality"""
    
    def setup_method(self):
        self.registry = FactoryRegistry()
    
    def test_singleton_pattern(self):
        """Test that FactoryRegistry is a singleton"""
        registry2 = FactoryRegistry()
        
        # Should be the same instance
        assert self.registry is registry2
    
    def test_register_and_get_factory(self):
        """Test factory registration and retrieval"""
        # Register factories
        self.registry.register('component', ComponentFactory())
        self.registry.register('client', ClientFactory())
        
        # Retrieve factories
        component_factory = self.registry.get_factory('component')
        client_factory = self.registry.get_factory('client')
        
        assert isinstance(component_factory, ComponentFactory)
        assert isinstance(client_factory, ClientFactory)
    
    def test_get_unregistered_factory(self):
        """Test getting unregistered factory"""
        with pytest.raises(FactoryError):
            self.registry.get_factory('unregistered')
    
    def test_list_registered_factories(self):
        """Test listing registered factories"""
        # Register some factories
        self.registry.register('factory1', ComponentFactory())
        self.registry.register('factory2', ClientFactory())
        
        factories = self.registry.list_registered_factories()
        
        assert 'factory1' in factories
        assert 'factory2' in factories
        assert isinstance(factories, list)
    
    def test_unregister_factory(self):
        """Test factory unregistration"""
        # Register and then unregister
        self.registry.register('temp_factory', ComponentFactory())
        self.registry.unregister('temp_factory')
        
        # Should fail to get after unregistration
        with pytest.raises(FactoryError):
            self.registry.get_factory('temp_factory')
    
    def test_clear_all_factories(self):
        """Test clearing all factories"""
        # Register some factories
        self.registry.register('factory1', ComponentFactory())
        self.registry.register('factory2', ClientFactory())
        
        # Clear all
        self.registry.clear_all()
        
        # Should have no factories
        factories = self.registry.list_registered_factories()
        assert len(factories) == 0
    
    def test_factory_validation(self):
        """Test factory validation"""
        # Test registering non-factory object
        with pytest.raises(FactoryError):
            self.registry.register('invalid', 'not_a_factory')
    
    def test_registry_integration(self):
        """Test integration with all factories"""
        # Register all factories
        self.registry.register('component', ComponentFactory())
        self.registry.register('client', ClientFactory())
        self.registry.register('server', ServerFactory())
        self.registry.register('dataset', DatasetFactory())
        
        # Verify all are registered
        factories = self.registry.list_registered_factories()
        
        assert 'component' in factories
        assert 'client' in factories
        assert 'server' in factories
        assert 'dataset' in factories
        
        # Verify we can get all of them
        component_factory = self.registry.get_factory('component')
        client_factory = self.registry.get_factory('client')
        server_factory = self.registry.get_factory('server')
        dataset_factory = self.registry.get_factory('dataset')
        
        assert isinstance(component_factory, ComponentFactory)
        assert isinstance(client_factory, ClientFactory)
        assert isinstance(server_factory, ServerFactory)
        assert isinstance(dataset_factory, DatasetFactory)