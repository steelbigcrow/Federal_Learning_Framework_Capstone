# --- 兜底：确保 from src... 可导入 ---
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# ------------------------------------------------

"""
Final model evaluation script.

This script evaluates trained federated learning models (both standard and LoRA)
on test datasets and generates visualization plots. It now uses the modular
evaluation framework from src.evaluation.
"""

import argparse
from pathlib import Path

import torch

from src.evaluation import ModelEvaluator


def main():
    parser = argparse.ArgumentParser("Evaluate ONE federated global checkpoint")
    parser.add_argument("--arch-config", required=True, help="Architecture configuration file")
    parser.add_argument("--checkpoint", required=True, help="server/round_N.pth (LoRA时为基模)")
    parser.add_argument("--lora-ckpt", help="LoRA 适配器（可选）")
    parser.add_argument("--outdir", required=True, help="Output directory for evaluation results")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mis-limit", type=int, default=36, help="Max misclassified examples to show")
    args = parser.parse_args()

    # Create evaluator
    evaluator = ModelEvaluator(
        arch_config_path=args.arch_config,
        device=args.device
    )
    
    # Run evaluation
    try:
        results = evaluator.evaluate_from_checkpoint(
            checkpoint_path=args.checkpoint,
            lora_checkpoint_path=args.lora_ckpt,
            output_dir=args.outdir,
            misclassified_limit=args.mis_limit
        )
        
        print(f"Evaluation completed successfully: {args.outdir}")
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())