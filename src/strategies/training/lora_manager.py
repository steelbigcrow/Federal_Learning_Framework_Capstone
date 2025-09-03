"""
LoRA management for federated learning.

This module provides OOP-based LoRA functionality,
replacing the legacy lora_utils.py functions.
"""

from typing import Iterable, List, Dict, Optional, Any
import torch
import torch.nn as nn
import loralib as lora
import os
from pathlib import Path

from ...core.base.component import FederatedComponent
from ...utils.serialization import save_checkpoint, load_checkpoint


class LoRAManager(FederatedComponent):
    """
    Manages LoRA (Low-Rank Adaptation) operations for federated learning.
    
    This class encapsulates all LoRA-related operations including
    module injection, parameter management, and checkpoint handling.
    """
    
    def __init__(
        self,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
        target_modules: Optional[Iterable[str]] = None
    ):
        """
        Initialize the LoRA manager.
        
        Args:
            r: LoRA rank
            alpha: LoRA scaling factor
            dropout: LoRA dropout rate
            target_modules: Target module types for LoRA injection
        """
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.dropout = dropout
        self.target_modules = target_modules or ["Linear", "Embedding"]
        
    def _replace_modules_with_lora(
        self,
        module: nn.Module,
        prefix: str,
        replaced: List[str]
    ) -> None:
        """
        Recursively replace modules with LoRA versions.
        
        Args:
            module: Module to process
            prefix: Current module prefix
            replaced: List of replaced module names
        """
        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            cls_name = child.__class__.__name__
            
            # Replace Linear layers
            if isinstance(child, nn.Linear) and (cls_name in self.target_modules or 'Linear' in self.target_modules):
                new_layer = lora.Linear(
                    child.in_features,
                    child.out_features,
                    r=self.r,
                    lora_alpha=self.alpha,
                    lora_dropout=self.dropout,
                    bias=child.bias is not None
                )
                new_layer.weight.data.copy_(child.weight.data)
                if child.bias is not None:
                    new_layer.bias.data.copy_(child.bias.data)
                setattr(module, name, new_layer)
                replaced.append(full_name)
                self.logger.debug(f"Replaced Linear layer: {full_name}")
                
            # Replace Embedding layers
            elif isinstance(child, nn.Embedding) and (cls_name in self.target_modules or 'Embedding' in self.target_modules):
                new_layer = lora.Embedding(
                    child.num_embeddings,
                    child.embedding_dim,
                    r=self.r,
                    lora_alpha=self.alpha,
                    max_norm=child.max_norm,
                    norm_type=child.norm_type,
                    scale_grad_by_freq=child.scale_grad_by_freq,
                    sparse=child.sparse,
                    padding_idx=child.padding_idx
                )
                new_layer.weight.data.copy_(child.weight.data)
                setattr(module, name, new_layer)
                replaced.append(full_name)
                self.logger.debug(f"Replaced Embedding layer: {full_name}")
                
            else:
                # Recurse into child modules
                self._replace_modules_with_lora(child, full_name, replaced)
                
    def inject_lora_modules(self, model: nn.Module) -> List[str]:
        """
        Inject LoRA into specified module types.
        
        Args:
            model: Model to inject LoRA into
            
        Returns:
            List of replaced module names
        """
        replaced: List[str] = []
        self._replace_modules_with_lora(model, prefix="", replaced=replaced)
        self.logger.info(f"Injected LoRA into {len(replaced)} modules")
        return replaced
        
    def mark_only_lora_as_trainable(
        self,
        model: nn.Module,
        train_classifier_head: bool = True
    ) -> None:
        """
        Set model to only train LoRA parameters.
        
        Args:
            model: Model to configure
            train_classifier_head: Whether to also train classifier head
        """
        lora.mark_only_lora_as_trainable(model)
        
        if train_classifier_head:
            for name, module in model.named_modules():
                if name.endswith('classifier') or name.endswith('head'):
                    for p in module.parameters():
                        p.requires_grad = True
                    self.logger.debug(f"Enabled training for classifier: {name}")
                    
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        self.logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100.0 * trainable_params / total_params:.2f}%)")
        
    def extract_lora_state_dict(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """
        Extract all LoRA parameters from model.
        
        Args:
            model: Model containing LoRA parameters
            
        Returns:
            Dictionary of LoRA parameters
        """
        lora_state_dict = {}
        for name, module in model.named_modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                # LoRA layers
                lora_state_dict[f"{name}.lora_A"] = module.lora_A.detach().clone()
                lora_state_dict[f"{name}.lora_B"] = module.lora_B.detach().clone()
                if hasattr(module, 'lora_embedding_A'):
                    lora_state_dict[f"{name}.lora_embedding_A"] = module.lora_embedding_A.detach().clone()
                if hasattr(module, 'lora_embedding_B'):
                    lora_state_dict[f"{name}.lora_embedding_B"] = module.lora_embedding_B.detach().clone()
                    
        self.logger.debug(f"Extracted {len(lora_state_dict)} LoRA parameters")
        return lora_state_dict
        
    def load_lora_state_dict(
        self,
        model: nn.Module,
        lora_state_dict: Dict[str, torch.Tensor],
        strict: bool = False
    ) -> None:
        """
        Load LoRA weights into model.
        
        Args:
            model: Target model
            lora_state_dict: LoRA weights dictionary
            strict: Whether to use strict mode (error on missing keys)
        """
        loaded_count = 0
        for name, module in model.named_modules():
            lora_A_key = f"{name}.lora_A"
            lora_B_key = f"{name}.lora_B"
            
            if lora_A_key in lora_state_dict and hasattr(module, 'lora_A'):
                module.lora_A.data.copy_(lora_state_dict[lora_A_key])
                loaded_count += 1
                
            if lora_B_key in lora_state_dict and hasattr(module, 'lora_B'):
                module.lora_B.data.copy_(lora_state_dict[lora_B_key])
                loaded_count += 1
                
            # Handle embedding LoRA
            lora_emb_A_key = f"{name}.lora_embedding_A"
            lora_emb_B_key = f"{name}.lora_embedding_B"
            
            if lora_emb_A_key in lora_state_dict and hasattr(module, 'lora_embedding_A'):
                module.lora_embedding_A.data.copy_(lora_state_dict[lora_emb_A_key])
                loaded_count += 1
                
            if lora_emb_B_key in lora_state_dict and hasattr(module, 'lora_embedding_B'):
                module.lora_embedding_B.data.copy_(lora_state_dict[lora_emb_B_key])
                loaded_count += 1
                
        if strict and loaded_count != len(lora_state_dict):
            raise ValueError(f"Strict mode: only loaded {loaded_count}/{len(lora_state_dict)} LoRA parameters")
            
        self.logger.info(f"Loaded {loaded_count} LoRA parameters")
        
    def save_lora_checkpoint(
        self,
        model: nn.Module,
        path: str,
        meta: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save LoRA checkpoint.
        
        Args:
            model: Model containing LoRA parameters
            path: Save path
            meta: Optional metadata
        """
        lora_state_dict = self.extract_lora_state_dict(model)
        save_checkpoint(lora_state_dict, path, meta)
        self.logger.info(f"Saved LoRA checkpoint to {path}")
        
    def load_lora_checkpoint(
        self,
        model: nn.Module,
        path: str,
        device: str = 'cpu',
        strict: bool = False
    ) -> None:
        """
        Load LoRA checkpoint.
        
        Args:
            model: Target model
            path: Checkpoint path
            device: Device to load to
            strict: Whether to use strict mode
        """
        checkpoint = load_checkpoint(path, device)
        if 'state_dict' in checkpoint:
            lora_state_dict = checkpoint['state_dict']
        else:
            lora_state_dict = checkpoint
            
        self.load_lora_state_dict(model, lora_state_dict, strict)
        self.logger.info(f"Loaded LoRA checkpoint from {path}")
        
    def load_base_model_checkpoint(
        self,
        model: nn.Module,
        path: str,
        device: str = 'cpu'
    ) -> None:
        """
        Load base model checkpoint (non-LoRA parameters).
        
        Args:
            model: Target model
            path: Checkpoint path
            device: Device to load to
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Base model checkpoint not found: {path}")
            
        checkpoint = load_checkpoint(path, device)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        # Load only non-LoRA parameters
        model_state = model.state_dict()
        filtered_state = {}
        
        for key in state_dict:
            if 'lora' not in key.lower() and key in model_state:
                filtered_state[key] = state_dict[key]
                
        model.load_state_dict(filtered_state, strict=False)
        self.logger.info(f"Loaded base model from {path} ({len(filtered_state)} parameters)")