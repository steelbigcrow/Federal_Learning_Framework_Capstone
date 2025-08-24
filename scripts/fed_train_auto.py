# --- 兜底：保证 from src... 在任何工作目录都能导入 ---
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# --------------------------------------------------------

"""
一条命令：训练 -> 自动评测（无需再手动敲第二条命令）
用法与 scripts/fed_train.py 一致，只是把文件名换成 fed_train_auto.py

例：
python scripts/fed_train_auto.py --arch-config configs/arch/mnist_mlp.yaml --train-config configs/federated.yaml --override run_name=mnist_mlp_base federated.num_rounds=5 --device cuda
python scripts/fed_train_auto.py --arch-config configs/arch/imdb_transformer.yaml --train-config configs/federated.yaml --use-lora --override run_name=tx_lora federated.num_rounds=5 --device cuda
"""

import argparse
import subprocess
import time
import re
from pathlib import Path

try:
    import yaml
except Exception:
    yaml = None

try:
    import torch
    _HAS_CUDA = torch.cuda.is_available()
except Exception:
    _HAS_CUDA = False


def _all_round_ckpts(server_dir: Path, is_lora: bool):
    patt = re.compile(rf"{'lora_' if is_lora else ''}round_(\d+)\.pth$")
    items = []
    if not server_dir.exists():
        return items
    for p in server_dir.glob("*.pth"):
        m = patt.fullmatch(p.name)
        if not m:
            continue
        items.append((int(m.group(1)), p))
    items.sort(key=lambda x: x[0])
    return items  # [(round, path), ...]


def _pick_ckpt(outputs_root: Path, run_name: str | None, num_rounds: int, is_lora: bool, started_at: float | None):
    """
    选择评测用的 checkpoint。
    优先顺序：
      1) 在包含 run_name 的目录里找 目标轮
      2) 在包含 run_name 的目录里找 最新一轮
      3) 全局扫描 目标轮（过滤掉训练前的旧文件）
      4) 全局扫描 最新一轮（过滤掉训练前的旧文件）
    """
    root = outputs_root / ("loras" if is_lora else "checkpoints")
    target = f"server/{'lora_' if is_lora else ''}round_{num_rounds}.pth"

    def latest_in(dirpath: Path):
        pairs = _all_round_ckpts(dirpath / "server", is_lora)
        return pairs[-1][1] if pairs else None

    cand_dirs = []
    if run_name:
        for d in root.glob(f"*{run_name}*"):
            if (d / "server").exists():
                cand_dirs.append(d)
        cand_dirs.sort(key=lambda p: p.stat().st_mtime)

    for d in reversed(cand_dirs):
        p = d / target
        if p.exists():
            return p, f"[in-run {d.name}] exact round_{num_rounds}"
    for d in reversed(cand_dirs):
        p = latest_in(d)
        if p is not None:
            return p, f"[in-run {d.name}] latest available"

    cands = []
    for d in root.glob("*"):
        if not (d / "server").exists():
            continue
        exact = d / target
        if exact.exists() and (started_at is None or exact.stat().st_mtime >= started_at - 5):
            cands.append((exact.stat().st_mtime, exact, f"[any-run {d.name}] exact round_{num_rounds}"))
        else:
            latest = latest_in(d)
            if latest and (started_at is None or latest.stat().st_mtime >= started_at - 5):
                cands.append((latest.stat().st_mtime, latest, f"[any-run {d.name}] latest available"))

    if not cands:
        return None, "no checkpoint found under outputs/*/{server}"
    cands.sort(key=lambda x: x[0])
    return cands[-1][1], cands[-1][2]


def _auto_eval_after_training(arch_config: str, train_config: str, use_lora_flag: bool | None,
                              viz_name: str | None, started_at: float | None, device: str | None):
    cfg = {}
    if yaml:
        with open(train_config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    outputs_root = Path((cfg.get("logging") or {}).get("root") or "./outputs")
    num_rounds = int((cfg.get("federated") or {}).get("num_rounds", 5))
    use_lora = bool(use_lora_flag) or bool(cfg.get("use_lora"))
    run_name = cfg.get("run_name") or Path(arch_config).stem
    viz_dir = outputs_root / "viz" / (viz_name or run_name)
    viz_dir.mkdir(parents=True, exist_ok=True)

    ckpt, why = _pick_ckpt(outputs_root, run_name, num_rounds, is_lora=use_lora, started_at=started_at)
    if ckpt is None:
        print(f"[auto-eval] 找不到融合后的全局权重（{why}）。跳过评测。")
        return

    print(f"[auto-eval] FOUND: {ckpt}  ({why})")

    cmd = [
        sys.executable, "scripts/eval_final.py",
        "--arch-config", arch_config,
        "--outdir", str(viz_dir),
    ]
    if device:
        cmd += ["--device", device]

    if use_lora:
        base_model_path = (cfg.get("lora") or {}).get("base_model_path")
        if not base_model_path:
            print("[auto-eval] federated.yaml 缺少 lora.base_model_path，跳过评测。")
            return
        base_ckpt = Path(base_model_path)
        if not base_ckpt.is_absolute():
            base_ckpt = outputs_root / base_model_path
        if not base_ckpt.exists():
            print(f"[auto-eval] 基模不存在：{base_ckpt}，跳过评测。")
            return
        cmd += ["--checkpoint", str(base_ckpt), "--lora-ckpt", str(ckpt)]
    else:
        cmd += ["--checkpoint", str(ckpt)]

    print("[auto-eval] RUN:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
        print(f"[auto-eval] 完成，结果在：{viz_dir}")
    except subprocess.CalledProcessError as e:
        print(f"[auto-eval] 评测失败：{e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch-config", required=True)
    ap.add_argument("--train-config", required=True)
    ap.add_argument("--override", nargs="*")
    ap.add_argument("--use-lora", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--data-cache-dir")
    ap.add_argument("--viz-name")
    ap.add_argument("--device", default=("cuda" if _HAS_CUDA else "cpu"))
    args = ap.parse_args()

    # 1) 跑原训练脚本
    started_at = time.time()
    cmd = [sys.executable, "scripts/fed_train.py",
           "--arch-config", args.arch_config,
           "--train-config", args.train_config]
    if args.override:
        cmd += ["--override"] + args.override
    if args.use_lora:
        cmd += ["--use-lora"]
    if args.no_cache:
        cmd += ["--no-cache"]
    if args.data_cache_dir:
        cmd += ["--data-cache-dir", args.data_cache_dir]

    print("[train] RUN:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    # 2) 自动评测
    _auto_eval_after_training(
        arch_config=args.arch_config,
        train_config=args.train_config,
        use_lora_flag=args.use_lora,
        viz_name=args.viz_name,
        started_at=started_at,
        device=args.device,
    )


if __name__ == "__main__":
    main()
