"""
Core model evaluation functionality.

This module provides the ModelEvaluator class which handles the complete
evaluation workflow including model loading, evaluation, and visualization.
"""

import json
import os
import re
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn

try:
    import yaml
except Exception:
    yaml = None

from .loaders import get_test_loader
from .metrics import evaluate_model
from .visualizer import visualize_results


class ModelEvaluator:
    """Handles model evaluation for both standard and LoRA models."""
    
    def __init__(self, device: str = None, outputs_root: str = "./outputs"):
        """
        Initialize model evaluator with minimal parameters.
        
        Args:
            device: Device to run evaluation on (auto-detect if None)
            outputs_root: Root directory for outputs
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.outputs_root = Path(outputs_root)
        
        # These will be set when loading configs
        self.arch_config = None
        self.arch_config_path = None
        self.dataset_name = None
        self.model_name = None
    
    @classmethod
    def from_configs(cls, arch_config_path: str, train_config_path: str = None, **kwargs):
        """
        Factory method to create evaluator from configuration files.
        
        Args:
            arch_config_path: Path to architecture configuration file
            train_config_path: Path to training configuration file (optional)
            **kwargs: Additional arguments for evaluator initialization
            
        Returns:
            ModelEvaluator instance with loaded configurations
        """
        evaluator = cls(**kwargs)
        evaluator._load_arch_config(arch_config_path)
        return evaluator
    
    def _load_arch_config(self, arch_config_path: str):
        """Load and validate architecture configuration."""
        self.arch_config_path = Path(arch_config_path)
        self.arch_config = self._read_yaml(self.arch_config_path)
        self.dataset_name = (self.arch_config.get("dataset") or "").lower()
        self.model_name = (self.arch_config.get("model") or "").lower()
        
        if not self.dataset_name or not self.model_name:
            raise ValueError(f"Architecture config missing dataset/model: {self.arch_config_path}")
    
    def evaluate(self, checkpoint_path: str, arch_config_path: str = None, 
                lora_checkpoint_path: str = None, output_dir: str = None, **kwargs) -> Dict[str, Any]:
        """
        Unified evaluation interface that handles all evaluation scenarios.
        
        Args:
            checkpoint_path: Path to main model checkpoint
            arch_config_path: Path to architecture config (required if not set via from_configs)
            lora_checkpoint_path: Path to LoRA adapter checkpoint (optional)
            output_dir: Directory to save evaluation results (auto-generate if None)
            **kwargs: Additional arguments for evaluation
            
        Returns:
            Dictionary containing evaluation results
        """
        # Load arch config if not already loaded
        if self.arch_config is None:
            if arch_config_path is None:
                raise ValueError("Architecture config must be provided either via from_configs() or arch_config_path parameter")
            self._load_arch_config(arch_config_path)
        
        return self.evaluate_from_checkpoint(checkpoint_path, lora_checkpoint_path, output_dir, **kwargs)
    
    def evaluate_from_checkpoint(self, checkpoint_path: str, lora_checkpoint_path: str = None,
                                output_dir: str = None, **kwargs) -> Dict[str, Any]:
        """
        Evaluate model from checkpoint files (requires arch_config to be loaded).
        
        Args:
            checkpoint_path: Path to main model checkpoint
            lora_checkpoint_path: Path to LoRA adapter checkpoint (optional)
            output_dir: Directory to save evaluation results (auto-generate if None)
            **kwargs: Additional arguments for evaluation
            
        Returns:
            Dictionary containing evaluation results
        """
        if self.arch_config is None:
            raise RuntimeError("Architecture config must be loaded first. Use evaluate() method or from_configs() factory.")
        
        # Create model
        model = self._build_model()
        
        # Load checkpoint(s)
        if lora_checkpoint_path:
            self._load_lora_model(model, checkpoint_path, lora_checkpoint_path)
        else:
            self._load_standard_model(model, checkpoint_path)
        
        # Create output directory if not specified
        if output_dir is None:
            output_dir = self._get_default_output_dir(checkpoint_path, lora_checkpoint_path)
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get test data loader
        test_loader = self._create_test_loader(checkpoint_path)
        
        # Evaluate model
        results = evaluate_model(model, test_loader, self.dataset_name, self.device)
        
        # Create visualizations
        visualize_results(results, output_path, **kwargs)
        
        # Print results
        self._print_results(results, output_path)
        
        return results
    
    def auto_evaluate_after_training(self, train_config_path: str, arch_config_path: str = None,
                                   use_lora: bool = False, run_name: str = None, started_at: float = None) -> bool:
        """
        Automatically evaluate model after training completion.
        
        Args:
            train_config_path: Path to training configuration file
            arch_config_path: Path to architecture config (required if not set via from_configs)
            use_lora: Whether LoRA was used in training
            run_name: Name of training run
            started_at: Training start timestamp for filtering checkpoints
            
        Returns:
            True if evaluation completed successfully, False otherwise
        """
        # Load arch config if not already loaded
        if self.arch_config is None:
            if arch_config_path is None:
                raise ValueError("Architecture config must be provided either via from_configs() or arch_config_path parameter")
            self._load_arch_config(arch_config_path)
        
        # Load training config
        train_config = self._read_yaml(Path(train_config_path))
        
        # Find checkpoint to evaluate
        checkpoint_info = self._find_evaluation_checkpoint(
            train_config, use_lora, run_name, started_at
        )
        
        if not checkpoint_info:
            print("[auto-eval] No suitable checkpoint found for evaluation.")
            return False
        
        checkpoint_path, lora_path, output_dir = checkpoint_info
        
        try:
            # Perform evaluation
            self.evaluate_from_checkpoint(
                checkpoint_path=str(checkpoint_path),
                lora_checkpoint_path=str(lora_path) if lora_path else None,
                output_dir=str(output_dir)
            )
            print(f"[auto-eval] Evaluation completed: {output_dir}")
            return True
        except Exception as e:
            print(f"[auto-eval] Evaluation failed: {e}")
            return False
    
    def _build_model(self):
        """Build model from architecture configuration."""
        from src.models import create_model
        
        extra = {}
        if self.dataset_name == "imdb":
            # Use defaults, actual values will be loaded from checkpoint
            extra = {
                "vocab_size": int(self.arch_config.get("vocab_size", 30000)),
                "pad_idx": int(self.arch_config.get("pad_idx", 1))
            }
        
        return create_model(self.dataset_name, self.model_name, self.arch_config, extra=extra)
    
    def _load_standard_model(self, model: nn.Module, checkpoint_path: str):
        """Load standard model checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        obj = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        
        state_dict = obj.get("state_dict") if isinstance(obj, dict) else obj
        if state_dict is None:
            state_dict = obj
        
        # Handle embedding size mismatch for text models
        self._maybe_resize_text_embedding(model, state_dict)
        
        # Filter compatible parameters
        kept = self._filter_state_to_model(model, state_dict)
        
        # Load parameters
        model.load_state_dict(kept, strict=False)
    
    def _load_lora_model(self, model: nn.Module, base_checkpoint_path: str, lora_checkpoint_path: str):
        """Load LoRA model (base + adapter)."""
        # First load base model
        self._load_standard_model(model, base_checkpoint_path)
        
        # Then load LoRA adapter
        lora_path = Path(lora_checkpoint_path)
        obj = torch.load(lora_path, map_location="cpu", weights_only=False)
        
        lora_state_dict = obj.get("state_dict") if isinstance(obj, dict) else obj
        if lora_state_dict is None:
            lora_state_dict = obj
        
        # Load LoRA parameters
        kept = self._filter_state_to_model(model, lora_state_dict)
        model.load_state_dict(kept, strict=False)
    
    def _create_test_loader(self, checkpoint_path: str):
        """Create test data loader with appropriate parameters."""
        loader_kwargs = {}
        
        if self.dataset_name == "imdb":
            # Try to get parameters from checkpoint sidecar
            sidecar_data = self._read_sidecar_json(Path(checkpoint_path))
            if sidecar_data:
                model_info = sidecar_data.get("model_info", {})
                model_config = model_info.get("model_config", {})
                loader_kwargs["pad_idx"] = int(model_config.get("pad_idx", 1))
            else:
                # Fall back to config defaults
                loader_kwargs["pad_idx"] = int(self.arch_config.get("pad_idx", 1))
        
        return get_test_loader(self.dataset_name, **loader_kwargs)
    
    def _find_evaluation_checkpoint(self, train_config: Dict, use_lora: bool, 
                                  run_name: str, started_at: float) -> Optional[Tuple[Path, Optional[Path], Path]]:
        """
        Find appropriate checkpoint for evaluation.
        
        Returns:
            Tuple of (checkpoint_path, lora_path, output_dir) or None if not found
        """
        num_rounds = int((train_config.get("federated") or {}).get("num_rounds", 5))
        run_name = run_name or self.arch_config_path.stem
        
        # Find checkpoint
        checkpoint_path = self._find_checkpoint(use_lora, run_name, num_rounds, started_at)
        if not checkpoint_path:
            return None
        
        # Determine paths
        lora_path = None
        if use_lora:
            # For LoRA, we need both base model and LoRA adapter
            base_model_path = (train_config.get("lora") or {}).get("base_model_path")
            if not base_model_path:
                print("[auto-eval] Missing lora.base_model_path in training config.")
                return None
            
            base_path = Path(base_model_path)
            if not base_path.is_absolute():
                base_path = self.outputs_root / base_model_path
            
            if not base_path.exists():
                print(f"[auto-eval] Base model not found: {base_path}")
                return None
            
            lora_path = checkpoint_path  # This is the LoRA checkpoint
            checkpoint_path = base_path  # This becomes the base model path
        
        # Create output directory in plots structure
        # For LoRA: use LoRA checkpoint path to determine directory
        # For standard: use standard checkpoint path
        if use_lora:
            lora_checkpoint_parent = lora_path.parent.parent.parent  # From server/lora_round_N.pth back to model_dir
            model_run_dir = lora_checkpoint_parent.name
        else:
            checkpoint_parent = checkpoint_path.parent.parent.parent  # From server/round_N.pth back to model_dir  
            model_run_dir = checkpoint_parent.name
        
        output_dir = self.outputs_root / ("loras" if use_lora else "models") / model_run_dir / "plots" / "server"
        
        return checkpoint_path, lora_path, output_dir
    
    def _find_checkpoint(self, is_lora: bool, run_name: str, num_rounds: int, 
                        started_at: float) -> Optional[Path]:
        """Find the most appropriate checkpoint for evaluation."""
        root = self.outputs_root / ("loras" if is_lora else "models")
        target_pattern = f"weights/server/{'lora_' if is_lora else ''}round_{num_rounds}.pth"
        
        # Look for exact round in run-specific directories
        candidate_dirs = []
        for d in root.glob(f"*{run_name}*"):
            if (d / "weights" / "server").exists():
                candidate_dirs.append(d)
        candidate_dirs.sort(key=lambda p: p.stat().st_mtime)
        
        # Try exact round first
        for d in reversed(candidate_dirs):
            exact_path = d / target_pattern
            if exact_path.exists():
                return exact_path
        
        # Try latest available round
        for d in reversed(candidate_dirs):
            available = self._get_all_round_checkpoints(d / "weights" / "server", is_lora)
            if available:
                return available[-1][1]  # Return latest
        
        # Global search as fallback
        candidates = []
        for d in root.glob("*"):
            if not (d / "weights" / "server").exists():
                continue
            
            exact_path = d / target_pattern
            if exact_path.exists() and (started_at is None or exact_path.stat().st_mtime >= started_at - 5):
                candidates.append((exact_path.stat().st_mtime, exact_path))
        
        if candidates:
            candidates.sort(key=lambda x: x[0])
            return candidates[-1][1]
        
        return None
    
    def _get_all_round_checkpoints(self, server_dir: Path, is_lora: bool):
        """Get all round checkpoints in a server directory."""
        pattern = re.compile(rf"{'lora_' if is_lora else ''}round_(\d+)\.pth$")
        items = []
        
        if not server_dir.exists():
            return items
        
        for p in server_dir.glob("*.pth"):
            match = pattern.fullmatch(p.name)
            if match:
                items.append((int(match.group(1)), p))
        
        items.sort(key=lambda x: x[0])
        return items
    
    def _get_default_output_dir(self, checkpoint_path: str, lora_path: str = None) -> str:
        """Generate default output directory based on checkpoint path."""
        ckpt_path = Path(lora_path if lora_path else checkpoint_path)
        model_run_dir = ckpt_path.parent.parent.name
        return str(self.outputs_root / "plots" / model_run_dir / "server")
    
    def _print_results(self, results: Dict[str, Any], output_path: Path):
        """Print evaluation results."""
        dataset = results["dataset"]
        
        if dataset == "mnist":
            accuracy = results["accuracy"]
            print(f"[MNIST] acc={accuracy:.4f} → {output_path}")
        elif dataset == "imdb":
            roc_auc = results["roc_auc"]
            pr_auc = results["pr_auc"]
            print(f"[IMDB] ROC-AUC={roc_auc:.4f} PR-AUC={pr_auc:.4f} → {output_path}")
    
    # Helper methods
    def _read_yaml(self, path: Path) -> Dict:
        """Read YAML file safely."""
        if not yaml:
            return {}
        try:
            return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:
            return {}
    
    def _read_sidecar_json(self, checkpoint_path: Path):
        """Read sidecar JSON file for checkpoint metadata."""
        json_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".json")
        if json_path.exists():
            try:
                return json.loads(json_path.read_text(encoding="utf-8"))
            except Exception:
                return None
        return None
    
    def _filter_state_to_model(self, model: nn.Module, state_dict: Dict) -> Dict:
        """Filter state dict to only include compatible parameters."""
        kept = {}
        model_state_dict = model.state_dict()
        
        for key, value in state_dict.items():
            if key in model_state_dict and model_state_dict[key].shape == value.shape:
                kept[key] = value
        
        return kept
    
    def _maybe_resize_text_embedding(self, model: nn.Module, state_dict: Dict):
        """Resize model embedding if checkpoint has different vocabulary size."""
        # Find embedding weight key
        embedding_key = None
        for key in state_dict.keys():
            if key.endswith("embedding.weight"):
                embedding_key = key
                break
        
        if embedding_key is None:
            return
        
        if not hasattr(model, "embedding") or not isinstance(model.embedding, nn.Embedding):
            return
        
        ckpt_vocab, ckpt_dim = state_dict[embedding_key].shape
        cur_vocab, cur_dim = model.embedding.weight.shape
        
        if ckpt_dim != cur_dim:
            return
        
        if ckpt_vocab != cur_vocab:
            pad_idx = getattr(model, "pad_idx", model.embedding.padding_idx)
            new_embedding = nn.Embedding(ckpt_vocab, ckpt_dim, padding_idx=pad_idx)
            new_embedding = new_embedding.to(
                model.embedding.weight.device, 
                dtype=model.embedding.weight.dtype
            )
            model.embedding = new_embedding