"""
AdaLoRA management for federated learning.

This module provides OOP-based AdaLoRA functionality,
replacing the legacy adalora_utils.py functions.
"""

import os
import sys
from typing import Iterable, List, Dict, Optional, Any, Tuple
import torch
import torch.nn as nn
from pathlib import Path

from ...core.base.component import FederatedComponent
from ...utils.serialization import save_checkpoint, load_checkpoint

# Add the AdaLoRA library to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'adaloralib'))

try:
    from loralib.adalora import SVDLinear, RankAllocator, compute_orth_regu
    from loralib.utils import mark_only_lora_as_trainable as adalora_mark_trainable
    from loralib.utils import lora_state_dict as adalora_state_dict
    from loralib import Embedding as LoRAEmbedding
    ADALORA_AVAILABLE = True
except ImportError:
    # Fallback implementations
    SVDLinear = None
    RankAllocator = None
    compute_orth_regu = None
    adalora_mark_trainable = None
    adalora_state_dict = None
    LoRAEmbedding = None
    ADALORA_AVAILABLE = False


class AdaLoRAManager(FederatedComponent):
    """
    Manages AdaLoRA (Adaptive Budget Allocation) operations for federated learning.
    
    This class encapsulates all AdaLoRA-related operations including
    module injection, rank allocation, and checkpoint handling.
    """
    
    def __init__(
        self,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
        target_modules: Optional[Iterable[str]] = None,
        budget: int = 100,
        beta1: float = 0.85,
        beta2: float = 0.85,
        orth_reg_weight: float = 0.1
    ):
        """
        Initialize the AdaLoRA manager.
        
        Args:
            r: Initial LoRA rank
            alpha: LoRA scaling factor
            dropout: LoRA dropout rate
            target_modules: Target module types for AdaLoRA injection
            budget: Total rank budget for AdaLoRA
            beta1: Exponential moving average coefficient for sensitivity
            beta2: Exponential moving average coefficient for uncertainty
            orth_reg_weight: Orthogonal regularization weight
        """
        super().__init__()
        
        if not ADALORA_AVAILABLE:
            raise ImportError(
                "AdaLoRA library not available. Please install from "
                "https://github.com/QingruZhang/AdaLoRA"
            )
            
        self.r = r
        self.alpha = alpha
        self.dropout = dropout
        self.target_modules = target_modules or ["Linear", "Embedding"]
        self.budget = budget
        self.beta1 = beta1
        self.beta2 = beta2
        self.orth_reg_weight = orth_reg_weight
        
        self.rank_allocator = None
        
    def _replace_modules_with_adalora(
        self,
        module: nn.Module,
        prefix: str,
        replaced: List[str]
    ) -> None:
        """
        Recursively replace modules with AdaLoRA versions.
        
        Args:
            module: Module to process
            prefix: Current module prefix
            replaced: List of replaced module names
        """
        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            cls_name = child.__class__.__name__
            
            # Replace Linear layers with SVDLinear
            if isinstance(child, nn.Linear) and (cls_name in self.target_modules or 'Linear' in self.target_modules):
                new_layer = SVDLinear(
                    child.in_features,
                    child.out_features,
                    r=self.r,
                    lora_alpha=self.alpha,
                    lora_dropout=self.dropout,
                    bias=child.bias is not None
                )
                # Copy weights
                new_layer.weight.data.copy_(child.weight.data)
                if child.bias is not None:
                    new_layer.bias.data.copy_(child.bias.data)
                setattr(module, name, new_layer)
                replaced.append(full_name)
                self.logger.debug(f"Replaced Linear with SVDLinear: {full_name}")
                
            # Replace Embedding layers (use standard LoRA for embeddings)
            elif isinstance(child, nn.Embedding) and (cls_name in self.target_modules or 'Embedding' in self.target_modules):
                new_layer = LoRAEmbedding(
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
                self.logger.debug(f"Replaced Embedding with LoRAEmbedding: {full_name}")
                
            else:
                # Recurse into child modules
                self._replace_modules_with_adalora(child, full_name, replaced)
                
    def inject_adalora_modules(self, model: nn.Module) -> List[str]:
        """
        Inject AdaLoRA into specified module types.
        
        Args:
            model: Model to inject AdaLoRA into
            
        Returns:
            List of replaced module names
        """
        replaced: List[str] = []
        self._replace_modules_with_adalora(model, prefix="", replaced=replaced)
        self.logger.info(f"Injected AdaLoRA into {len(replaced)} modules")
        return replaced
        
    def mark_only_adalora_as_trainable(
        self,
        model: nn.Module,
        train_classifier_head: bool = True
    ) -> None:
        """
        Set model to only train AdaLoRA parameters.
        
        Args:
            model: Model to configure
            train_classifier_head: Whether to also train classifier head
        """
        # Use AdaLoRA's marking function
        adalora_mark_trainable(model)
        
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
        
    def create_rank_allocator(
        self,
        model: nn.Module,
        update_freq: int = 200
    ) -> 'RankAllocator':
        """
        Create rank allocator for AdaLoRA.
        
        Args:
            model: Model with AdaLoRA modules
            update_freq: Frequency of rank updates
            
        Returns:
            RankAllocator instance
        """
        self.rank_allocator = RankAllocator(
            model=model,
            lora_r=self.r,
            target_rank=self.budget,
            init_warmup=500,
            final_warmup=1500,
            mask_interval=50,
            beta1=self.beta1,
            beta2=self.beta2,
            total_step=update_freq * 10
        )
        
        self.logger.info(f"Created rank allocator with budget {self.budget}")
        return self.rank_allocator
        
    def extract_adalora_state_dict(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """
        Extract all AdaLoRA parameters from model.
        
        Args:
            model: Model containing AdaLoRA parameters
            
        Returns:
            Dictionary of AdaLoRA parameters
        """
        state_dict = adalora_state_dict(model, bias='none')
        self.logger.debug(f"Extracted {len(state_dict)} AdaLoRA parameters")
        return state_dict
        
    def load_adalora_state_dict(
        self,
        model: nn.Module,
        adalora_state_dict: Dict[str, torch.Tensor],
        strict: bool = False
    ) -> None:
        """
        Load AdaLoRA weights into model.
        
        Args:
            model: Target model
            adalora_state_dict: AdaLoRA weights dictionary
            strict: Whether to use strict mode
        """
        missing, unexpected = model.load_state_dict(adalora_state_dict, strict=False)
        
        if strict and (missing or unexpected):
            raise ValueError(f"Strict mode: missing keys {missing}, unexpected keys {unexpected}")
            
        self.logger.info(f"Loaded AdaLoRA state dict with {len(adalora_state_dict)} parameters")
        
    def save_adalora_checkpoint(
        self,
        model: nn.Module,
        path: str,
        meta: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save AdaLoRA checkpoint.
        
        Args:
            model: Model containing AdaLoRA parameters
            path: Save path
            meta: Optional metadata
        """
        adalora_state_dict = self.extract_adalora_state_dict(model)
        
        # Add rank allocator state if available
        if self.rank_allocator is not None:
            checkpoint_data = {
                'state_dict': adalora_state_dict,
                'rank_allocator': self.rank_allocator.state_dict(),
                'meta': meta or {}
            }
        else:
            checkpoint_data = {
                'state_dict': adalora_state_dict,
                'meta': meta or {}
            }
            
        save_checkpoint(checkpoint_data, path)
        self.logger.info(f"Saved AdaLoRA checkpoint to {path}")
        
    def load_adalora_checkpoint(
        self,
        model: nn.Module,
        path: str,
        device: str = 'cpu',
        strict: bool = False
    ) -> None:
        """
        Load AdaLoRA checkpoint.
        
        Args:
            model: Target model
            path: Checkpoint path
            device: Device to load to
            strict: Whether to use strict mode
        """
        checkpoint = load_checkpoint(path, device)
        
        if 'state_dict' in checkpoint:
            adalora_state_dict = checkpoint['state_dict']
        else:
            adalora_state_dict = checkpoint
            
        self.load_adalora_state_dict(model, adalora_state_dict, strict)
        
        # Load rank allocator if available
        if 'rank_allocator' in checkpoint and self.rank_allocator is not None:
            self.rank_allocator.load_state_dict(checkpoint['rank_allocator'])
            self.logger.info("Loaded rank allocator state")
            
        self.logger.info(f"Loaded AdaLoRA checkpoint from {path}")
        
    def get_adalora_regularization_loss(self, model: nn.Module) -> torch.Tensor:
        """
        Get AdaLoRA orthogonal regularization loss.
        
        Args:
            model: Model with AdaLoRA modules
            
        Returns:
            Regularization loss tensor
        """
        reg_loss = torch.tensor(0.0, device=next(model.parameters()).device)
        
        for module in model.modules():
            if isinstance(module, SVDLinear):
                reg_loss += compute_orth_regu(model, regu_weight=self.orth_reg_weight)
                break  # compute_orth_regu already handles all SVDLinear modules
                
        return reg_loss
        
    def get_adalora_parameter_stats(self, model: nn.Module) -> Dict[str, Any]:
        """
        Get statistics about AdaLoRA parameters.
        
        Args:
            model: Model with AdaLoRA modules
            
        Returns:
            Dictionary of parameter statistics
        """
        stats = {
            'total_adalora_params': 0,
            'active_ranks': {},
            'total_budget': self.budget,
            'used_budget': 0
        }
        
        for name, module in model.named_modules():
            if isinstance(module, SVDLinear):
                # Get current rank
                if hasattr(module, 'ranknum'):
                    current_rank = module.ranknum.sum().item()
                    stats['active_ranks'][name] = current_rank
                    stats['used_budget'] += current_rank
                    
                # Count parameters
                adalora_params = (
                    module.lora_A.numel() + 
                    module.lora_B.numel() + 
                    module.lora_E.numel()
                )
                stats['total_adalora_params'] += adalora_params
                
        stats['budget_utilization'] = stats['used_budget'] / stats['total_budget'] if stats['total_budget'] > 0 else 0.0
        
        return stats