"""
Round-based evaluation system for federated learning.

This module provides automatic evaluation functionality that runs after each federated round,
replacing the previous --auto-eval flag system.
"""

import torch
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from .metrics import evaluate_model
from .visualizer import visualize_results
from .loaders import get_test_loader


class RoundEvaluator:
    """
    Handles automatic evaluation after each federated learning round.
    
    This class integrates evaluation and visualization functionality directly into
    the server training loop, eliminating the need for separate --auto-eval flag.
    """
    
    def __init__(self, model_info: Dict[str, Any], device: str = "cuda", 
                 lora_cfg: Optional[Dict] = None, adalora_cfg: Optional[Dict] = None):
        """
        Initialize round evaluator.
        
        Args:
            model_info: Dictionary containing model and dataset information
            device: Device to run evaluation on
            lora_cfg: LoRA configuration if applicable
            adalora_cfg: AdaLoRA configuration if applicable
        """
        self.model_info = model_info
        self.device = device
        self.lora_cfg = lora_cfg or {}
        self.adalora_cfg = adalora_cfg or {}
        
        # Initialize test loader
        self.test_loader = self._create_test_loader()
        
    def _create_test_loader(self):
        """Create test data loader for evaluation."""
        try:
            dataset_name = self.model_info.get('dataset', 'unknown')
            
            if dataset_name == 'mnist':
                return get_test_loader(dataset_name, batch_size=64)
            elif dataset_name == 'imdb':
                pad_idx = 1  # Default padding index
                max_len = self.model_info.get('max_seq_len', 256)
                return get_test_loader(dataset_name, pad_idx=pad_idx, max_len=max_len, batch_size=64)
            else:
                print(f"[Warning] Unknown dataset type: {dataset_name}, evaluation disabled")
                return None
                
        except Exception as e:
            print(f"[Error] Failed to create test loader: {e}")
            return None
    
    def evaluate_round(self, model: torch.nn.Module, round_num: int, 
                  paths_manager) -> Dict[str, Any]:
        """
        Evaluate model after a federated round and save to standard metrics/plots directories.
        
        Args:
            model: The global model to evaluate
            round_num: Current round number
            paths_manager: PathManager instance for consistent directory structure
            
        Returns:
            Dictionary containing evaluation results
        """
        if self.test_loader is None:
            print(f"[Round {round_num}] Skipping server evaluation: no test data available")
            return {}
            
        try:
            print(f"[Round {round_num}] Starting server-side evaluation...")
            
            # Perform evaluation
            dataset_name = self.model_info.get('dataset', 'unknown')
            results = evaluate_model(
                model=model,
                data_loader=self.test_loader,
                dataset_name=dataset_name,
                device=self.device
            )
            
            # Build server evaluation metrics in the expected format
            server_metrics = {
                "round": round_num,
                "timestamp": datetime.now().isoformat(),
                "dataset": dataset_name,
                "server_evaluation": {
                    "acc": results.get('accuracy', 0.0),
                    "loss": results.get('loss', 0.0),  # Use actual loss from evaluate_model
                    "f1": results.get('f1_score', 0.0)  # Use actual F1 score from evaluate_model
                }
            }
            
            # Save server metrics to metrics/server/server_round_x_metrics.json
            # Convert string path to Path object
            server_metrics_path = Path(paths_manager.server_round_metrics(round_num))
            server_metrics_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Rename to follow the required naming convention
            server_metrics_file = server_metrics_path.parent / f"server_round_{round_num}_metrics.json"
            
            with open(server_metrics_file, 'w') as f:
                json.dump(server_metrics, f, indent=2, default=str)
                
            # Generate server evaluation plot
            self._generate_server_plot(server_metrics, round_num, paths_manager)
            
            print(f"[Round {round_num}] Server evaluation completed: test_acc={results.get('accuracy', 0):.4f}, test_loss={results.get('loss', 0):.4f}, f1={results.get('f1_score', 0):.4f}")
            print(f"[Round {round_num}] Server metrics saved: {server_metrics_file}")
            
            return server_metrics
            
        except Exception as e:
            print(f"[Round {round_num}] Server evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def _generate_server_plot(self, server_metrics: Dict[str, Any], round_num: int, paths_manager):
        """Generate server evaluation plot for a specific round."""
        try:
            import matplotlib.pyplot as plt
            
            # Create plots directory - convert string to Path object
            plots_server_dir = Path(paths_manager.plots_server_dir)
            plots_server_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract metrics for plotting
            eval_data = server_metrics.get("server_evaluation", {})
            acc = eval_data.get("acc", 0.0)
            loss = eval_data.get("loss", 0.0)
            f1 = eval_data.get("f1", 0.0)
            
            # Create a simple bar chart for this round's metrics
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
            
            # Accuracy plot - expanded y-axis range [0, 1.2]
            ax1.bar(['Accuracy'], [acc], color='blue', alpha=0.7)
            ax1.set_ylim(0, 1.2)
            ax1.set_ylabel('Score')
            ax1.set_title(f'Round {round_num} - Test Accuracy')
            ax1.text(0, acc + 0.02, f'{acc:.4f}', ha='center', va='bottom')
            
            # Loss plot - expanded y-axis with dynamic upper limit
            max_loss_display = max(1.2, loss * 1.3)  # At least 1.2, or 30% above the loss value
            ax2.bar(['Loss'], [loss], color='red', alpha=0.7)
            ax2.set_ylim(0, max_loss_display)
            ax2.set_ylabel('Loss')
            ax2.set_title(f'Round {round_num} - Test Loss')
            ax2.text(0, loss + max(0.02, loss*0.1), f'{loss:.4f}', ha='center', va='bottom')
            
            # F1 plot - expanded y-axis range [0, 1.2]
            ax3.bar(['F1 Score'], [f1], color='green', alpha=0.7)
            ax3.set_ylim(0, 1.2)
            ax3.set_ylabel('Score')
            ax3.set_title(f'Round {round_num} - F1 Score')
            ax3.text(0, f1 + 0.02, f'{f1:.4f}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            # Save plot with the required naming convention
            plot_file = plots_server_dir / f"server_round_{round_num}_metrics.png"
            plt.savefig(plot_file, dpi=160, bbox_inches='tight')
            plt.close()
            
            print(f"[Round {round_num}] Server plot saved: {plot_file}")
            
        except Exception as e:
            print(f"[Round {round_num}] Server plot generation failed: {e}")
            import traceback
            traceback.print_exc()
    
    def generate_training_summary(self, paths_manager, total_rounds: int):
        """
        Generate summary plots for server evaluation across all rounds.
        
        Args:
            paths_manager: PathManager instance for directory access
            total_rounds: Total number of rounds completed
        """
        try:
            print("[Training Complete] Generating server evaluation summary...")
            
            # Collect all server round metrics - convert string to Path object
            metrics_server_dir = Path(paths_manager.metrics_server_dir)
            if not metrics_server_dir.exists():
                print("[Warning] No server metrics found for summary")
                return
                
            round_results = []
            for round_num in range(1, total_rounds + 1):
                metrics_file = metrics_server_dir / f"server_round_{round_num}_metrics.json"
                if metrics_file.exists():
                    with open(metrics_file, 'r') as f:
                        round_results.append(json.load(f))
            
            if not round_results:
                print("[Warning] No server evaluation results found for summary")
                return
            
            # Generate training progress plot
            self._plot_server_training_progress(round_results, paths_manager)
            
            print("[Training Complete] Server evaluation summary completed")
            
        except Exception as e:
            print(f"[Training Complete] Server evaluation summary failed: {e}")
            import traceback
            traceback.print_exc()
    
    def _plot_server_training_progress(self, round_results: list, paths_manager):
        """Plot server evaluation progress across rounds."""
        try:
            import matplotlib.pyplot as plt
            
            rounds = [r['round'] for r in round_results]
            server_eval = [r.get('server_evaluation', {}) for r in round_results]
            accuracies = [eval_data.get('acc', 0) for eval_data in server_eval]
            losses = [eval_data.get('loss', 0) for eval_data in server_eval]
            f1_scores = [eval_data.get('f1', 0) for eval_data in server_eval]
            
            # Create plots
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
            
            # Accuracy progress
            ax1.plot(rounds, accuracies, 'b-o', linewidth=2, markersize=6)
            ax1.set_xlabel('Round')
            ax1.set_ylabel('Test Accuracy')
            ax1.set_title('Server Test Accuracy Progress')
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 1)
            
            # Loss progress
            ax2.plot(rounds, losses, 'r-o', linewidth=2, markersize=6)
            ax2.set_xlabel('Round')
            ax2.set_ylabel('Test Loss')
            ax2.set_title('Server Test Loss Progress')
            ax2.grid(True, alpha=0.3)
            
            # F1 score progress
            ax3.plot(rounds, f1_scores, 'g-o', linewidth=2, markersize=6)
            ax3.set_xlabel('Round')
            ax3.set_ylabel('F1 Score')
            ax3.set_title('Server F1 Score Progress')
            ax3.grid(True, alpha=0.3)
            ax3.set_ylim(0, 1)
            
            plt.tight_layout()
            
            # Save plot to server plots directory - convert string to Path object
            plots_server_dir = Path(paths_manager.plots_server_dir)
            plots_server_dir.mkdir(parents=True, exist_ok=True)
            progress_plot = plots_server_dir / "server_training_progress.png"
            plt.savefig(progress_plot, dpi=160, bbox_inches='tight')
            plt.close()
            
            print(f"[Training Complete] Server progress plot saved: {progress_plot}")
            
        except Exception as e:
            print(f"[Training Complete] Server progress plot failed: {e}")
            import traceback
            traceback.print_exc()