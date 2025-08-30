"""
Evaluation metrics computation functions.

This module provides functions for computing various evaluation metrics
for MNIST and IMDB datasets.
"""

import torch
import numpy as np
from sklearn.metrics import (
    confusion_matrix, roc_curve, precision_recall_curve,
    roc_auc_score, average_precision_score
)


@torch.no_grad()
def evaluate_mnist_model(model, data_loader, device="cuda"):
    """
    Evaluate a model on MNIST test data.
    
    Args:
        model: PyTorch model to evaluate
        data_loader: DataLoader for MNIST test data
        device: Device to run evaluation on
        
    Returns:
        tuple: (accuracy, confusion_matrix, true_labels, predictions)
    """
    model.to(device).eval()
    
    all_labels = []
    all_predictions = []
    
    for batch in data_loader:
        logits = model(batch["x"].to(device))
        predictions = logits.argmax(1).cpu()
        labels = batch["y"].cpu()
        
        all_predictions.append(predictions)
        all_labels.append(labels)
    
    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_predictions).numpy()
    
    # Calculate metrics
    accuracy = (y_true == y_pred).mean()
    cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))
    
    return accuracy, cm, y_true, y_pred


@torch.no_grad() 
def evaluate_imdb_model(model, data_loader, device="cuda"):
    """
    Evaluate a model on IMDB test data.
    
    Args:
        model: PyTorch model to evaluate
        data_loader: DataLoader for IMDB test data
        device: Device to run evaluation on
        
    Returns:
        tuple: (roc_auc, pr_auc, roc_data, pr_data)
    """
    model.to(device).eval()
    
    all_labels = []
    all_probabilities = []
    
    for batch in data_loader:
        logits = model(batch["x"].to(device))
        # Get probability of positive class (class 1)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        labels = batch["y"].numpy()
        
        all_probabilities.extend(probs.tolist())
        all_labels.extend(labels.tolist())
    
    y_true = np.asarray(all_labels)
    y_prob = np.asarray(all_probabilities)
    
    # Calculate metrics
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    
    # Get curve data
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    
    return roc_auc, pr_auc, (fpr, tpr), (precision, recall)


def evaluate_model(model, data_loader, dataset_name, device="cuda"):
    """
    Generic model evaluation function.
    
    Args:
        model: PyTorch model to evaluate
        data_loader: Test data loader
        dataset_name: Name of dataset ('mnist' or 'imdb')
        device: Device to run evaluation on
        
    Returns:
        Dictionary containing evaluation results
        
    Raises:
        ValueError: If dataset_name is not supported
    """
    dataset_name = dataset_name.lower()
    
    if dataset_name == "mnist":
        accuracy, cm, y_true, y_pred = evaluate_mnist_model(model, data_loader, device)
        return {
            "dataset": "mnist",
            "accuracy": float(accuracy),
            "confusion_matrix": cm,
            "true_labels": y_true,
            "predictions": y_pred
        }
    elif dataset_name == "imdb":
        roc_auc, pr_auc, roc_data, pr_data = evaluate_imdb_model(model, data_loader, device)
        return {
            "dataset": "imdb", 
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
            "roc_data": roc_data,
            "pr_data": pr_data
        }
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")