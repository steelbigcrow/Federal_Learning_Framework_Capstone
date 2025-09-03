"""
Logging management for federated learning.

This module provides OOP-based logging functionality,
replacing the legacy logging_utils.py functions.
"""

import json
import os
from typing import Dict, Any, Optional
from pathlib import Path

from ..base.component import FederatedComponent


class LogManager(FederatedComponent):
    """
    Manages logging operations for federated learning.
    
    This class encapsulates all logging-related operations including
    metrics logging and text logging.
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize the log manager.
        
        Args:
            output_dir: Base output directory for logs
        """
        self.output_dir = output_dir
        super().__init__()

        
    def _validate_config(self) -> None:
        """
        Validate log manager configuration.
        
        LogManager doesn't require specific configuration validation.
        """
        pass
        
    def _initialize(self) -> None:
        """
        Initialize the log manager.
        
        Ensure output directory exists if specified.
        """
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
        
    def write_metrics_json(self, path: str, metrics: Dict[str, Any]) -> None:
        """
        Write metrics data to a JSON file.
        
        Args:
            path: File save path
            metrics: Metrics dictionary to save
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        self.logger.debug(f"Metrics written to {path}")
        
    def write_text_log(self, path: str, text: str) -> None:
        """
        Write text log to a file (append mode).
        
        Args:
            path: File save path
            text: Text content to write
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'a', encoding='utf-8') as f:
            f.write(text + "\n")
        self.logger.debug(f"Text log appended to {path}")
        
    def log_training_metrics(
        self,
        metrics: Dict[str, Any],
        round_num: int,
        client_id: Optional[int] = None,
        is_server: bool = False
    ) -> None:
        """
        Log training metrics with automatic path generation.
        
        Args:
            metrics: Metrics to log
            round_num: Current round number
            client_id: Client ID (None for server)
            is_server: Whether these are server metrics
        """
        if self.output_dir:
            if is_server:
                path = self.output_dir / "metrics" / "server" / f"round_{round_num}.json"
            else:
                path = self.output_dir / "metrics" / "clients" / f"client_{client_id}" / f"round_{round_num}.json"
                
            self.write_metrics_json(str(path), metrics)
            
    def log_event(
        self,
        event: str,
        round_num: Optional[int] = None,
        client_id: Optional[int] = None,
        is_server: bool = False
    ) -> None:
        """
        Log an event with automatic path generation.
        
        Args:
            event: Event text to log
            round_num: Current round number
            client_id: Client ID (None for server)
            is_server: Whether this is a server event
        """
        if self.output_dir:
            if is_server:
                path = self.output_dir / "logs" / "server" / "events.log"
            else:
                path = self.output_dir / "logs" / "clients" / f"client_{client_id}" / "events.log"
                
            # Add round number to event if provided
            if round_num is not None:
                event = f"[Round {round_num}] {event}"
                
            self.write_text_log(str(path), event)