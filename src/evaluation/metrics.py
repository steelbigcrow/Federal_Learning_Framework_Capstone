"""
Evaluation metrics computation functions.

This module provides functions for computing various evaluation metrics
for MNIST and IMDB datasets.
"""

import torch
import numpy as np
from sklearn.metrics import confusion_matrix


@torch.no_grad()
def evaluate_mnist_model(model, data_loader, device="cuda"):
    """
    Evaluate a model on MNIST test data.
    
    Args:
        model: PyTorch model to evaluate
        data_loader: DataLoader for MNIST test data
        device: Device to run evaluation on
        
    Returns:
        tuple: (accuracy, confusion_matrix, true_labels, predictions, loss, f1_score)
    """
    from sklearn.metrics import f1_score
    import torch.nn.functional as F
    
    model.to(device).eval()
    
    all_labels = []
    all_predictions = []
    total_loss = 0.0
    num_batches = 0
    
    for batch in data_loader:
        inputs = batch["x"].to(device)
        labels = batch["y"].to(device)
        
        logits = model(inputs)
        
        # Calculate loss
        loss = F.cross_entropy(logits, labels)
        total_loss += loss.item()
        num_batches += 1
        
        predictions = logits.argmax(1).cpu()
        labels = labels.cpu()
        
        all_predictions.append(predictions)
        all_labels.append(labels)
    
    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_predictions).numpy()
    
    # Calculate metrics
    accuracy = (y_true == y_pred).mean()
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    f1 = f1_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))
    
    return accuracy, cm, y_true, y_pred, avg_loss, f1


@torch.no_grad() 
def evaluate_imdb_model(model, data_loader, device="cuda"):
    """
    Evaluate a model on IMDB test data.
    
    Args:
        model: PyTorch model to evaluate
        data_loader: DataLoader for IMDB test data
        device: Device to run evaluation on
        
    Returns:
        dict: Evaluation metrics (accuracy, loss, f1_score)
    """
    from sklearn.metrics import f1_score
    import torch.nn.functional as F
    
    model.to(device).eval()
    
    all_labels = []
    all_predictions = []
    total_loss = 0.0
    num_batches = 0
    
    for batch in data_loader:
        inputs = batch["x"].to(device)
        labels = batch["y"].to(device)
        
        logits = model(inputs)
        
        # Calculate loss
        loss = F.cross_entropy(logits, labels)
        total_loss += loss.item()
        num_batches += 1
        
        # Get predictions
        predictions = torch.argmax(logits, dim=1).cpu().numpy()
        labels = labels.cpu().numpy()
        
        all_predictions.extend(predictions.tolist())
        all_labels.extend(labels.tolist())
    
    y_true = np.asarray(all_labels)
    y_pred = np.asarray(all_predictions)
    
    # Calculate metrics
    accuracy = (y_true == y_pred).mean()
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    f1 = f1_score(y_true, y_pred, average='binary')  # Binary classification for IMDB
    
    return {"accuracy": accuracy, "loss": avg_loss, "f1_score": f1}


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
        accuracy, cm, y_true, y_pred, loss, f1 = evaluate_mnist_model(model, data_loader, device)
        return {
            "dataset": "mnist",
            "accuracy": float(accuracy),
            "loss": float(loss),
            "f1_score": float(f1),
            "confusion_matrix": cm,
            "true_labels": y_true,
            "predictions": y_pred
        }
    elif dataset_name == "imdb":
        metrics = evaluate_imdb_model(model, data_loader, device)
        return {
            "dataset": "imdb", 
            "accuracy": float(metrics["accuracy"]),
            "loss": float(metrics["loss"]),
            "f1_score": float(metrics["f1_score"])
        }
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")