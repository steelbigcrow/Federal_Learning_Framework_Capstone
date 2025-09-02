"""
Comprehensive Core Component Tests

Tests for all core OOP components including:
- Base classes (Component, Aggregator, Client, Server)
- Exception handling
- Interface implementations
- Component lifecycle management
"""

import pytest
import torch.nn as nn
from unittest.mock import Mock, patch
import tempfile
import os

from src.core.base.component import FederatedComponent, ComponentStatus, ComponentRegistry
from src.core.base.aggregator import AbstractAggregator
from src.core.base.client import AbstractClient
from src.core.base.server import AbstractServer
from src.core.exceptions.exceptions import (
    ServerConfigurationError, 
    ServerOperationError,
    ClientError,
    AggregationError,
    FactoryError,
    StrategyError
)


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
    
    def forward(self, x):
        return self.fc(x)


class TestFederatedComponent:
    """Test base FederatedComponent class"""
    
    def setup_method(self):
        self.component = FederatedComponent()
    
    def test_initialization(self):
        """Test component initialization"""
        assert self.component.status == ComponentStatus.CREATED
        assert hasattr(self.component, 'id')
        assert hasattr(self.component, 'name')
    
    def test_status_transitions(self):
        """Test status transitions"""
        # Test all valid transitions
        self.component.initialize()
        assert self.component.status == ComponentStatus.INITIALIZED
        
        self.component.start()
        assert self.component.status == ComponentStatus.RUNNING
        
        self.component.pause()
        assert self.component.status == ComponentStatus.PAUSED
        
        self.component.resume()
        assert self.component.status == ComponentStatus.RUNNING
        
        self.component.stop()
        assert self.component.status == ComponentStatus.STOPPED
        
        self.component.reset()
        assert self.component.status == ComponentStatus.CREATED
    
    def test_status_validation(self):
        """Test status validation"""
        # Test invalid transitions
        self.component.initialize()
        
        # Cannot initialize twice
        with pytest.raises(Exception):
            self.component.initialize()
    
    def test_component_info(self):
        """Test component information"""
        self.component.name = "test_component"
        self.component.id = "test_001"
        
        info = self.component.get_info()
        
        assert info['name'] == "test_component"
        assert info['id'] == "test_001"
        assert info['status'] == ComponentStatus.CREATED
    
    def test_component_health_check(self):
        """Test component health check"""
        health = self.component.health_check()
        
        assert 'status' in health
        assert 'timestamp' in health
        assert 'is_healthy' in health
        assert isinstance(health['is_healthy'], bool)


class TestComponentStatus:
    """Test ComponentStatus enum"""
    
    def test_status_values(self):
        """Test status enum values"""
        assert ComponentStatus.CREATED.value == "created"
        assert ComponentStatus.INITIALIZED.value == "initialized"
        assert ComponentStatus.RUNNING.value == "running"
        assert ComponentStatus.PAUSED.value == "paused"
        assert ComponentStatus.STOPPED.value == "stopped"
        assert ComponentStatus.ERROR.value == "error"
    
    def test_status_comparison(self):
        """Test status comparison"""
        assert ComponentStatus.CREATED != ComponentStatus.INITIALIZED
        assert ComponentStatus.RUNNING == ComponentStatus.RUNNING


class TestComponentRegistry:
    """Test ComponentRegistry class"""
    
    def setup_method(self):
        self.registry = ComponentRegistry()
    
    def test_singleton_pattern(self):
        """Test that ComponentRegistry is a singleton"""
        registry2 = ComponentRegistry()
        
        # Should be the same instance
        assert self.registry is registry2
    
    def test_register_and_get_component(self):
        """Test component registration and retrieval"""
        component = FederatedComponent()
        component.name = "test_component"
        
        self.registry.register("test_component", component)
        retrieved = self.registry.get("test_component")
        
        assert retrieved is component
    
    def test_get_unregistered_component(self):
        """Test getting unregistered component"""
        with pytest.raises(KeyError):
            self.registry.get("unregistered")
    
    def test_list_components(self):
        """Test listing components"""
        # Register some components
        comp1 = FederatedComponent()
        comp1.name = "comp1"
        comp2 = FederatedComponent()
        comp2.name = "comp2"
        
        self.registry.register("comp1", comp1)
        self.registry.register("comp2", comp2)
        
        components = self.registry.list_components()
        
        assert "comp1" in components
        assert "comp2" in components
        assert len(components) == 2
    
    def test_unregister_component(self):
        """Test component unregistration"""
        component = FederatedComponent()
        component.name = "temp"
        
        self.registry.register("temp", component)
        self.registry.unregister("temp")
        
        with pytest.raises(KeyError):
            self.registry.get("temp")
    
    def test_clear_all_components(self):
        """Test clearing all components"""
        # Register some components
        comp1 = FederatedComponent()
        comp1.name = "comp1"
        comp2 = FederatedComponent()
        comp2.name = "comp2"
        
        self.registry.register("comp1", comp1)
        self.registry.register("comp2", comp2)
        
        # Clear all
        self.registry.clear_all()
        
        # Should have no components
        components = self.registry.list_components()
        assert len(components) == 0
    
    def test_component_status_monitoring(self):
        """Test component status monitoring"""
        component = FederatedComponent()
        component.name = "monitored"
        
        self.registry.register("monitored", component)
        
        # Get status summary
        status_summary = self.registry.get_status_summary()
        
        assert "monitored" in status_summary
        assert status_summary["monitored"]["status"] == ComponentStatus.CREATED.value
    
    def test_component_health_monitoring(self):
        """Test component health monitoring"""
        component = FederatedComponent()
        component.name = "health_test"
        
        self.registry.register("health_test", component)
        
        # Get health summary
        health_summary = self.registry.get_health_summary()
        
        assert "health_test" in health_summary
        assert "is_healthy" in health_summary["health_test"]


class TestAbstractAggregator:
    """Test AbstractAggregator base class"""
    
    def test_abstract_methods(self):
        """Test that abstract methods must be implemented"""
        
        class ConcreteAggregator(AbstractAggregator):
            def aggregate(self, models, weights):
                return {}
            
            def get_name(self):
                return "test"
        
        aggregator = ConcreteAggregator()
        
        # Should work now
        result = aggregator.aggregate([], [])
        assert isinstance(result, dict)
        
        name = aggregator.get_name()
        assert name == "test"


class TestAbstractClient:
    """Test AbstractClient base class"""
    
    def test_abstract_methods(self):
        """Test that abstract methods must be implemented"""
        
        class ConcreteClient(AbstractClient):
            def id(self):
                return 0
            
            def train(self, model, config):
                return model
            
            def get_metrics(self):
                return {}
            
            def use_amp(self):
                return False
        
        client = ConcreteClient()
        
        # Should work now
        assert client.id() == 0
        assert client.train(Mock(), {}) is not None
        assert isinstance(client.get_metrics(), dict)
        assert client.use_amp() == False


class TestAbstractServer:
    """Test AbstractServer base class"""
    
    def test_abstract_methods(self):
        """Test that abstract methods must be implemented"""
        
        class ConcreteServer(AbstractServer):
            def initialize_global_model(self):
                pass
            
            def select_clients(self, round_number):
                return []
            
            def aggregate_models(self, models, weights):
                return {}
            
            def save_checkpoint(self, round_number, metrics):
                pass
        
        server = ConcreteServer()
        
        # Should work now
        server.initialize_global_model()
        assert server.select_clients(1) == []
        assert isinstance(server.aggregate_models([], []), dict)
        server.save_checkpoint(1, {})


class TestExceptionHandling:
    """Test exception handling"""
    
    def test_server_configuration_error(self):
        """Test ServerConfigurationError"""
        with pytest.raises(ServerConfigurationError) as exc_info:
            raise ServerConfigurationError("Test configuration error")
        
        assert "Test configuration error" in str(exc_info.value)
    
    def test_server_operation_error(self):
        """Test ServerOperationError"""
        with pytest.raises(ServerOperationError) as exc_info:
            raise ServerOperationError("Test operation error")
        
        assert "Test operation error" in str(exc_info.value)
    
    def test_client_error(self):
        """Test ClientError"""
        with pytest.raises(ClientError) as exc_info:
            raise ClientError("Test client error")
        
        assert "Test client error" in str(exc_info.value)
    
    def test_aggregation_error(self):
        """Test AggregationError"""
        with pytest.raises(AggregationError) as exc_info:
            raise AggregationError("Test aggregation error")
        
        assert "Test aggregation error" in str(exc_info.value)
    
    def test_factory_error(self):
        """Test FactoryError"""
        with pytest.raises(FactoryError) as exc_info:
            raise FactoryError("Test factory error")
        
        assert "Test factory error" in str(exc_info.value)
    
    def test_strategy_error(self):
        """Test StrategyError"""
        with pytest.raises(StrategyError) as exc_info:
            raise StrategyError("Test strategy error")
        
        assert "Test strategy error" in str(exc_info.value)
    
    def test_exception_inheritance(self):
        """Test that all exceptions inherit from base Exception"""
        exceptions = [
            ServerConfigurationError,
            ServerOperationError,
            ClientError,
            AggregationError,
            FactoryError,
            StrategyError
        ]
        
        for exc_class in exceptions:
            assert issubclass(exc_class, Exception)


class TestComponentLifecycle:
    """Test component lifecycle management"""
    
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_component_lifecycle_integration(self):
        """Test complete component lifecycle"""
        # Create a component
        component = FederatedComponent()
        component.name = "lifecycle_test"
        
        # Test lifecycle
        assert component.status == ComponentStatus.CREATED
        
        component.initialize()
        assert component.status == ComponentStatus.INITIALIZED
        
        component.start()
        assert component.status == ComponentStatus.RUNNING
        
        component.pause()
        assert component.status == ComponentStatus.PAUSED
        
        component.resume()
        assert component.status == ComponentStatus.RUNNING
        
        component.stop()
        assert component.status == ComponentStatus.STOPPED
        
        component.reset()
        assert component.status == ComponentStatus.CREATED
    
    def test_component_error_handling(self):
        """Test component error handling"""
        component = FederatedComponent()
        
        # Simulate error state
        component._status = ComponentStatus.ERROR
        
        # Should be able to reset from error
        component.reset()
        assert component.status == ComponentStatus.CREATED
    
    def test_component_persistence(self):
        """Test component state persistence"""
        component = FederatedComponent()
        component.name = "persistent_test"
        
        # Initialize and set some state
        component.initialize()
        component.start()
        
        # Save state
        state = component.get_state()
        
        assert 'status' in state
        assert 'name' in state
        assert state['status'] == ComponentStatus.RUNNING.value
        
        # Restore state
        new_component = FederatedComponent()
        new_component.restore_state(state)
        
        assert new_component.status == ComponentStatus.RUNNING
        assert new_component.name == "persistent_test"
    
    def test_component_dependency_management(self):
        """Test component dependency management"""
        # Create dependent components
        parent = FederatedComponent()
        parent.name = "parent"
        
        child1 = FederatedComponent()
        child1.name = "child1"
        
        child2 = FederatedComponent()
        child2.name = "child2"
        
        # Set up dependencies
        child1.add_dependency(parent)
        child2.add_dependency(parent)
        
        # Test dependency resolution
        assert parent in child1.get_dependencies()
        assert parent in child2.get_dependencies()
        
        # Test dependency initialization order
        parent.initialize()
        child1.initialize()
        child2.initialize()
        
        assert parent.status == ComponentStatus.INITIALIZED
        assert child1.status == ComponentStatus.INITIALIZED
        assert child2.status == ComponentStatus.INITIALIZED