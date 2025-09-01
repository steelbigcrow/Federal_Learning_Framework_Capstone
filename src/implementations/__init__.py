"""
Implementations module for the federated learning framework.

This module contains concrete implementations of the abstract classes defined in the core module.
The implementations use the OOP design patterns while maintaining backward compatibility with existing code.
"""

from .clients.federated_client import FederatedClient
from .servers.federated_server import FederatedServer
from .aggregators.federated_aggregator import FederatedAggregator

__all__ = [
    'FederatedClient',
    'FederatedServer', 
    'FederatedAggregator'
]