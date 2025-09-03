"""
Checkpoint management for federated learning.

This module provides OOP-based checkpoint management functionality,
replacing the legacy checkpoints.py functions.
"""

import os
from typing import Dict, Any, Optional
from pathlib import Path

from ...utils.serialization import save_checkpoint, load_checkpoint
from ..base.component import FederatedComponent


class CheckpointManager(FederatedComponent):
    """
    Manages checkpoint saving and loading operations for federated learning.
    
    This class encapsulates all checkpoint-related operations including
    saving and loading client/server model states.
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize the checkpoint manager.
        
        Args:
            output_dir: Base output directory for checkpoints
        """
        self.output_dir = output_dir
        super().__init__()

    
    def _validate_config(self) -> None:
        """
        Validate checkpoint manager configuration.
        
        CheckpointManager doesn't require specific configuration validation.
        """
        pass
        
    def _initialize(self) -> None:
        """
        Initialize the checkpoint manager.
        
        Ensure output directory exists if specified.
        """
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
        
    def save_client_checkpoint(
        self,
        state_dict: Dict[str, Any],
        path: str,
        meta: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save a client model checkpoint.
        
        Args:
            state_dict: Model state dictionary
            path: Save path for the checkpoint
            meta: Optional metadata to include
        """
        self.logger.debug(f"Saving client checkpoint to {path}")
        save_checkpoint(state_dict, path, meta)
        self.logger.info(f"Client checkpoint saved: {path}")
        
    def save_server_checkpoint(
        self,
        state_dict: Dict[str, Any],
        path: str,
        meta: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save a server/global model checkpoint.
        
        Args:
            state_dict: Model state dictionary
            path: Save path for the checkpoint
            meta: Optional metadata to include
        """
        self.logger.debug(f"Saving server checkpoint to {path}")
        save_checkpoint(state_dict, path, meta)
        self.logger.info(f"Server checkpoint saved: {path}")
        
    def load_checkpoint(
        self,
        path: str,
        device: str = 'cpu'
    ) -> Dict[str, Any]:
        """
        Load a checkpoint from disk.
        
        Args:
            path: Path to the checkpoint file
            device: Device to load the checkpoint to
            
        Returns:
            Loaded checkpoint dictionary
        """
        self.logger.debug(f"Loading checkpoint from {path}")
        checkpoint = load_checkpoint(path, device)
        self.logger.info(f"Checkpoint loaded: {path}")
        return checkpoint
        
    # Backward compatibility aliases
    def save_client_round(
        self,
        state_dict: Dict[str, Any],
        path: str,
        meta: Optional[Dict[str, Any]] = None
    ) -> None:
        """Backward compatibility wrapper for save_client_checkpoint."""
        self.save_client_checkpoint(state_dict, path, meta)
        
    def save_global_round(
        self,
        state_dict: Dict[str, Any],
        path: str,
        meta: Optional[Dict[str, Any]] = None
    ) -> None:
        """Backward compatibility wrapper for save_server_checkpoint."""
        self.save_server_checkpoint(state_dict, path, meta)