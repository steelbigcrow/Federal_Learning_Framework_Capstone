"""
Core federated learning components.

This module provides the main classes and aggregation functions for federated learning,
now fully implemented using the new OOP architecture.
"""

# Direct imports from OOP implementations
from ..implementations.clients.federated_client import FederatedClient
from ..implementations.servers.federated_server import FederatedServer
from ..implementations.aggregators import FederatedAggregator
from ..implementations.aggregators.federated_aggregator import FederatedAggregator
# 提取聚合函数作为模块级别的函数
fedavg = FederatedAggregator.fedavg
lora_fedavg = FederatedAggregator.lora_fedavg
adalora_fedavg = FederatedAggregator.adalora_fedavg
get_trainable_keys = FederatedAggregator.get_trainable_keys

# Aliases for backward compatibility
Client = FederatedClient
Server = FederatedServer

__all__ = [
    # Main classes
    'Client', 'Server', 'FederatedClient', 'FederatedServer', 'FederatedAggregator',
    # Aggregation functions
    'fedavg', 'lora_fedavg', 'adalora_fedavg', 'get_trainable_keys'
]