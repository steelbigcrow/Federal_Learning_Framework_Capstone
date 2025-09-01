"""
Core federated learning components.

This module provides the main classes and aggregation functions for federated learning,
including Server, Client implementations, and aggregation algorithms.

The module now includes both the original implementations and new OOP implementations
for backward compatibility during the refactoring process.
"""

# Original implementations (for backward compatibility)
from .server import Server
from .client import Client
from .aggregator import fedavg, lora_fedavg, adalora_fedavg, get_trainable_keys

# New OOP implementations
from ..implementations.clients import FederatedClient
from ..implementations.servers import FederatedServer
from ..implementations.aggregators import FederatedAggregator

__all__ = [
    # Original implementations
    'Server', 'Client', 'fedavg', 'lora_fedavg', 'adalora_fedavg', 'get_trainable_keys',
    # New OOP implementations
    'FederatedClient', 'FederatedServer', 'FederatedAggregator'
]