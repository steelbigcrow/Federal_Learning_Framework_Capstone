"""
Metrics visualization for federated learning.

This module provides OOP-based visualization functionality,
replacing the legacy plotting.py functions.
"""

import json
import os
from typing import Dict, List, Optional, Any
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments

from ..base.component import FederatedComponent


class MetricsVisualizer(FederatedComponent):
    """
    Manages metrics visualization for federated learning.
    
    This class encapsulates all plotting and visualization operations
    for client and server training metrics.
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize the metrics visualizer.
        
        Args:
            output_dir: Base output directory for plots
        """
        self.output_dir = output_dir
        super().__init__()

        
    def _validate_config(self) -> None:
        """
        Validate metrics visualizer configuration.
        
        MetricsVisualizer doesn't require specific configuration validation.
        """
        pass
        
    def _initialize(self) -> None:
        """
        Initialize the metrics visualizer.
        
        Ensure output directory exists if specified.
        """
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
        
    def plot_client_metrics(
        self,
        client_id: int,
        round_metrics_history: List[Dict[str, Any]],
        save_path: str,
        round_number: Optional[int] = None
    ) -> None:
        """
        Plot metrics for a specific client.
        
        Args:
            client_id: Client ID
            round_metrics_history: List of round metrics
            save_path: Path to save the plot
            round_number: Optional round number for filename
        """
        if not round_metrics_history:
            return
            
        # Ensure save directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Extract epoch data
        all_epochs = []
        train_acc = []
        train_f1 = []
        train_loss = []
        test_acc = []
        test_f1 = []
        
        epoch_counter = 0
        for round_idx, round_data in enumerate(round_metrics_history):
            # Check for epoch_history (new format)
            if 'epoch_history' in round_data and round_data['epoch_history']:
                for epoch_data in round_data['epoch_history']:
                    epoch_counter += 1
                    all_epochs.append(epoch_counter)
                    # Fix: epoch_history contains 'acc', 'f1', 'loss' not 'train_acc' etc.
                    train_acc.append(epoch_data.get('acc', 0))
                    train_f1.append(epoch_data.get('f1', 0))
                    train_loss.append(epoch_data.get('loss', 0))
                    # Test metrics come from the round data, not epoch history
                    test_acc.append(round_data.get('test_acc', 0))
                    test_f1.append(round_data.get('test_f1', 0))
            else:
                # Old format: single epoch per round
                epoch_counter += 1
                all_epochs.append(epoch_counter)
                train_acc.append(round_data.get('train_acc', 0))
                train_f1.append(round_data.get('train_f1', 0))
                train_loss.append(round_data.get('train_loss', 0))
                test_acc.append(round_data.get('test_acc', 0))
                test_f1.append(round_data.get('test_f1', 0))
        
        if not all_epochs:
            self.logger.warning(f"No epochs to plot for client {client_id}")
            return
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Client {client_id} Training Metrics', fontsize=16)
        
        # Plot training accuracy
        axes[0, 0].plot(all_epochs, train_acc, 'b-')
        axes[0, 0].set_title('Training Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].grid(True)
        
        # Plot training F1
        axes[0, 1].plot(all_epochs, train_f1, 'g-')
        axes[0, 1].set_title('Training F1 Score')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('F1 Score')
        axes[0, 1].grid(True)
        
        # Plot training loss
        axes[0, 2].plot(all_epochs, train_loss, 'r-')
        axes[0, 2].set_title('Training Loss')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Loss')
        axes[0, 2].grid(True)
        
        # Plot test accuracy
        axes[1, 0].plot(all_epochs, test_acc, 'b--')
        axes[1, 0].set_title('Test Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].grid(True)
        
        # Plot test F1
        axes[1, 1].plot(all_epochs, test_f1, 'g--')
        axes[1, 1].set_title('Test F1 Score')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('F1 Score')
        axes[1, 1].grid(True)
        
        # Hide the last subplot
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Client {client_id} metrics plot saved to {save_path}")
        
    def load_client_metrics_history(
        self,
        client_dir: str,
        num_rounds: int
    ) -> List[Dict[str, Any]]:
        """
        Load client metrics history from JSON files.
        
        Args:
            client_dir: Directory containing client metrics
            num_rounds: Number of rounds to load
            
        Returns:
            List of metrics dictionaries
        """
        metrics_history = []
        
        for round_num in range(1, num_rounds + 1):
            metrics_path = os.path.join(client_dir, f"round_{round_num}.json")
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    metrics_history.append(json.load(f))
            else:
                self.logger.warning(f"Metrics file not found: {metrics_path}")
                
        return metrics_history
        
    def plot_all_clients_metrics(
        self,
        client_ids: list,
        metrics_clients_dir: str,
        plots_dir: str,
        current_round: int
    ) -> None:
        """
        Plot metrics for all clients.
        
        Args:
            client_ids: List of client IDs to plot
            metrics_clients_dir: Directory containing all client metrics
            plots_dir: Directory to save plots
            current_round: Current training round
        """
        os.makedirs(plots_dir, exist_ok=True)
        
        for client_id in client_ids:
            client_metrics_dir = os.path.join(metrics_clients_dir, f"client_{client_id}")
            if os.path.exists(client_metrics_dir):
                metrics_history = self.load_client_metrics_history(client_metrics_dir, current_round)
                if metrics_history:
                    # 为每个客户端创建独立的子目录，符合CLAUDE.md规范
                    client_plots_dir = os.path.join(plots_dir, f"client_{client_id}")
                    os.makedirs(client_plots_dir, exist_ok=True)
                    # Fix: Include round number in filename to prevent overwriting
                    save_path = os.path.join(client_plots_dir, f"client_{client_id}_round_{current_round}_metrics.png")
                    self.plot_client_metrics(client_id, metrics_history, save_path, round_number=current_round)
            else:
                self.logger.warning(f"Client {client_id} metrics directory not found")
                
    def plot_server_metrics(
        self,
        server_metrics: Dict[str, List[float]],
        save_path: str
    ) -> None:
        """
        Plot server/global model metrics.
        
        Args:
            server_metrics: Dictionary of metric lists
            save_path: Path to save the plot
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Extract metrics
        rounds = list(range(1, len(server_metrics.get('accuracy', [])) + 1))
        accuracy = server_metrics.get('accuracy', [])
        f1_score = server_metrics.get('f1_score', [])
        loss = server_metrics.get('loss', [])
        
        # Create subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Server Model Metrics', fontsize=16)
        
        # Plot accuracy
        if accuracy:
            axes[0].plot(rounds, accuracy, 'b-o')
            axes[0].set_title('Global Model Accuracy')
            axes[0].set_xlabel('Round')
            axes[0].set_ylabel('Accuracy')
            axes[0].grid(True)
            
        # Plot F1 score
        if f1_score:
            axes[1].plot(rounds, f1_score, 'g-o')
            axes[1].set_title('Global Model F1 Score')
            axes[1].set_xlabel('Round')
            axes[1].set_ylabel('F1 Score')
            axes[1].grid(True)
            
        # Plot loss
        if loss:
            axes[2].plot(rounds, loss, 'r-o')
            axes[2].set_title('Global Model Loss')
            axes[2].set_xlabel('Round')
            axes[2].set_ylabel('Loss')
            axes[2].grid(True)
            
        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Server metrics plot saved to {save_path}")