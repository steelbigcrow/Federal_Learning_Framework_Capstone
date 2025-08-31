"""
Core federated learning components.

This module provides the main classes and aggregation functions for federated learning,
including Server, Client implementations, and aggregation algorithms.
"""

from .server import Server
from .client import Client
from .aggregator import fedavg, lora_fedavg

__all__ = ['Server', 'Client', 'fedavg', 'lora_fedavg']