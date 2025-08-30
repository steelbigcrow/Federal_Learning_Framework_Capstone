"""
Visualization functions for evaluation results.

This module provides functions to create and save various plots and charts
for visualizing model evaluation results.
"""

import math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from datasets import load_dataset


class ResultVisualizer:
    """Handles visualization of evaluation results."""
    
    def __init__(self, output_dir: Path):
        """
        Initialize visualizer with output directory.
        
        Args:
            output_dir: Directory to save visualization plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def visualize_mnist_results(self, accuracy, confusion_matrix, true_labels, predictions, misclassified_limit=36):
        """
        Create MNIST evaluation visualizations.
        
        Args:
            accuracy: Model accuracy
            confusion_matrix: Confusion matrix
            true_labels: Ground truth labels  
            predictions: Model predictions
            misclassified_limit: Maximum number of misclassified examples to show
        """
        # Confusion matrix plot
        self._plot_mnist_confusion_matrix(accuracy, confusion_matrix)
        
        # Per-class accuracy plot
        self._plot_mnist_per_class_accuracy(confusion_matrix)
        
        # Misclassified examples plot
        self._plot_mnist_misclassified(true_labels, predictions, misclassified_limit)
    
    def visualize_imdb_results(self, roc_auc, pr_auc, roc_data, pr_data):
        """
        Create IMDB evaluation visualizations.
        
        Args:
            roc_auc: ROC AUC score
            pr_auc: PR AUC score
            roc_data: Tuple of (fpr, tpr) for ROC curve
            pr_data: Tuple of (precision, recall) for PR curve
        """
        # ROC curve plot
        self._plot_imdb_roc_curve(roc_auc, roc_data)
        
        # PR curve plot  
        self._plot_imdb_pr_curve(pr_auc, pr_data)
    
    def _plot_mnist_confusion_matrix(self, accuracy, cm):
        """Plot MNIST confusion matrix."""
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(10)))
        disp.plot(values_format="d")
        plt.title(f"MNIST Confusion Matrix (acc={accuracy:.4f})")
        plt.tight_layout()
        plt.savefig(self.output_dir / "mnist_confusion_matrix.png", dpi=160)
        plt.close()
    
    def _plot_mnist_per_class_accuracy(self, cm):
        """Plot MNIST per-class accuracy."""
        per_class_acc = cm.diagonal() / cm.sum(1).clip(min=1)
        
        plt.figure()
        plt.bar(range(10), per_class_acc)
        plt.ylim(0, 1)
        plt.xlabel("Class")
        plt.ylabel("Accuracy")
        plt.title("Per-Class Accuracy")
        plt.tight_layout()
        plt.savefig(self.output_dir / "mnist_per_class_acc.png", dpi=160)
        plt.close()
    
    def _plot_mnist_misclassified(self, true_labels, predictions, limit):
        """Plot misclassified MNIST examples."""
        import numpy as np
        
        # Find misclassified indices
        misclassified_indices = np.where(true_labels != predictions)[0].astype(int)[:limit]
        
        if len(misclassified_indices) == 0:
            return
        
        # Load dataset for visualization
        ds = load_dataset("mnist", split="test")
        
        # Create subplot grid
        cols = 6
        rows = math.ceil(len(misclassified_indices) / cols)
        plt.figure(figsize=(cols * 2, rows * 2))
        
        for i, idx in enumerate(misclassified_indices):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(ds[int(idx)]["image"], cmap="gray")
            plt.axis("off")
            plt.title(f"gt:{int(true_labels[idx])} pred:{int(predictions[idx])}")
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "mnist_misclassified.png", dpi=160)
        plt.close()
    
    def _plot_imdb_roc_curve(self, roc_auc, roc_data):
        """Plot IMDB ROC curve."""
        fpr, tpr = roc_data
        
        plt.figure()
        plt.plot(fpr, tpr)
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title(f"ROC (AUC={roc_auc:.4f})")
        plt.tight_layout()
        plt.savefig(self.output_dir / "imdb_roc.png", dpi=160)
        plt.close()
    
    def _plot_imdb_pr_curve(self, pr_auc, pr_data):
        """Plot IMDB Precision-Recall curve."""
        precision, recall = pr_data
        
        plt.figure()
        plt.plot(recall, precision)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"PR (AUC={pr_auc:.4f})")
        plt.tight_layout()
        plt.savefig(self.output_dir / "imdb_pr.png", dpi=160)
        plt.close()


def visualize_results(results_dict, output_dir: Path, **kwargs):
    """
    Convenience function to visualize evaluation results.
    
    Args:
        results_dict: Dictionary containing evaluation results
        output_dir: Directory to save plots
        **kwargs: Additional arguments for specific visualizations
    """
    visualizer = ResultVisualizer(output_dir)
    dataset = results_dict["dataset"]
    
    if dataset == "mnist":
        visualizer.visualize_mnist_results(
            accuracy=results_dict["accuracy"],
            confusion_matrix=results_dict["confusion_matrix"], 
            true_labels=results_dict["true_labels"],
            predictions=results_dict["predictions"],
            misclassified_limit=kwargs.get("misclassified_limit", 36)
        )
    elif dataset == "imdb":
        visualizer.visualize_imdb_results(
            roc_auc=results_dict["roc_auc"],
            pr_auc=results_dict["pr_auc"],
            roc_data=results_dict["roc_data"],
            pr_data=results_dict["pr_data"]
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")