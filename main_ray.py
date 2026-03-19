#!/usr/bin/env python3
"""
main_ray.py – launch TF-JEPA pre-train + fine-tune on a Ray cluster

Usage examples
--------------
# default SleepEEG ➜ Epilepsy, 4 workers, GPUs:
uv run python main_ray.py --exp sleepeeg --num-workers 4

# cross-domain: SleepEEG ➜ Gesture, re-use an old encoder, 150 fine-tune epochs
uv run python main_ray.py --exp sleepeeg --tgt Gesture \
          --load-pretrained ./models/sleepeeg/jepa_pretrained_5e_0.0003lr.ckpt \
          --epochs-ft 150 --num-workers 2
"""

import argparse, pathlib
import ray
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig, RunConfig, FailureConfig, Checkpoint
import torch, tempfile, os

# ─────────────────────────────────────────────────────────────
# Local repo imports (assumes your package layout is unchanged)
# ─────────────────────────────────────────────────────────────
from distributed.pretrain_loop import pretrain_loop_per_worker
from distributed.finetune_loop import finetune_loop_per_worker
from distributed.ray_datasets import load_parquet_shard 
from configs import (
    EXPERIMENTS, TGT_INFO, to_dict, load_config
)

# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser("Ray launcher for TF-JEPA")
    p.add_argument("--exp", required=True,
                   help="har, ecg, fd, sleepeeg … or comma list / 'all'")
    p.add_argument("--src", default=None,
                   help="Override source dataset folder or keyword")
    p.add_argument("--tgt", default=None,
                   help="Override target dataset folder or keyword")
    p.add_argument("--epochs-pre", type=int, default=20)
    p.add_argument("--epochs-ft",  type=int, default=40)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=24)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--momentum", type=float, default=0.99)
    p.add_argument("--ckpt-freq", type=int, default=5,
                   help="Save a Ray Checkpoint every N epochs during pre-train")
    p.add_argument("--load-pretrained", default=None,
                   help="Local .ckpt path to bypass the Ray pre-train stage")
    p.add_argument("--use-gpu", action="store_true",
                   help="If set, each Ray worker grabs a GPU (default: CPU)")
    return p.parse_args()


def normalise(path_or_key: str) -> str:
    """Map bare keyword -> ./datasets/<key>, leave explicit paths untouched"""
    return path_or_key if ("/" in path_or_key or path_or_key.startswith(".")) \
        else f"./datasets/{path_or_key}"


def ckpt_file_to_ray(path: str) -> Checkpoint:
    state = torch.load(path, map_location="cpu")
    with tempfile.TemporaryDirectory() as tmp:
        file = os.path.join(tmp, "checkpoint.pkl")
        torch.save({"model_state": state}, file)   # Ray expects pickle dict
        return Checkpoint.from_directory(tmp)

# ─────────────────────────────────────────────────────────────
# Main driver
# ─────────────────────────────────────────────────────────────
def run_ray_experiment(exp_name: str, args):
    exp_cfg  = EXPERIMENTS[exp_name]
    cfg_obj  = load_config(exp_cfg["config_module"])
    cfg_dict = to_dict(cfg_obj)

    cfg_dict["lr"] = args.lr                      # CLI LR override

    # -------- resolve dataset folders ----------
    src_raw = args.src or exp_cfg["source_data"]
    tgt_raw = args.tgt or exp_cfg["target_data"]
    src_path, tgt_path = map(normalise, (src_raw, tgt_raw))

    # -------- adjust class count if needed ------
    tgt_key = pathlib.Path(tgt_path).stem
    if tgt_key in TGT_INFO:
        cfg_dict["num_classes_target"], cfg_dict["target_batch_size"] = TGT_INFO[tgt_key]

    # ───────── read Ray Datasets (Parquet must exist) ─────────
    train_ds    = load_parquet_shard(src_path, "train")       # <src>/train_ds.parquet
    finetune_ds = load_parquet_shard(tgt_path, "finetune")    # <tgt>/finetune_ds.parquet
    test_ds     = load_parquet_shard(tgt_path, "test")        # <tgt>/test_ds.parquet

    # ----------------------------------------------------------
    # 1) PRE-TRAIN  (unchanged)
    # ----------------------------------------------------------
    if args.load_pretrained:
        pre_ckpt = ckpt_file_to_ray(args.load_pretrained)   # ← Checkpoint object
        print(f"✓  Loaded external ckpt {args.load_pretrained}")
    else:
        pre_trainer = TorchTrainer(
            train_loop_per_worker=pretrain_loop_per_worker,
            train_loop_config=dict(
                seed=args.seed,
                num_epochs=args.epochs_pre,
                batch_size=args.batch_size,
                lr=args.lr,
                ckpt_freq=args.ckpt_freq,
                cfg_dict=cfg_dict,
                momentum=args.momentum,
            ),
            scaling_config=ScalingConfig(
                num_workers=args.num_workers,
                use_gpu=args.use_gpu,
            ),
            run_config=RunConfig(
                name=f"{exp_name}_pretrain",
                failure_config=FailureConfig(max_failures=3),
            ),
            datasets={"train": train_ds},
        )
        pre_result = pre_trainer.fit()
        pre_ckpt   = pre_result.checkpoint

    # ----------------------------------------------------------
    # 2) FINE-TUNE  (unchanged)
    # ----------------------------------------------------------
    fine_trainer = TorchTrainer(
        train_loop_per_worker=finetune_loop_per_worker,
        train_loop_config=dict(
            seed=args.seed,
            num_epochs=args.epochs_ft,
            batch_size=cfg_dict["target_batch_size"],
            lr=args.lr,
            cfg_dict=cfg_dict,
            pretrain_ckpt=pre_ckpt,
        ),
        scaling_config=ScalingConfig(
            num_workers=1,     # fine-tune usually single GPU
            use_gpu=args.use_gpu,
        ),
        run_config=RunConfig(
            name=f"{exp_name}_finetune",
            failure_config=FailureConfig(max_failures=3),
        ),
        datasets={"finetune": finetune_ds, "test": test_ds},
    )
    result = fine_trainer.fit()
    print(f"[{exp_name.upper()}] final metrics →", result.metrics)


def main():
    args = get_args()
    ray.init()        # will connect to an existing cluster in Databricks
    exp_list = (list(EXPERIMENTS.keys()) if args.exp.lower() == "all"
                else [e.strip() for e in args.exp.split(",")])
    for e in exp_list:
        run_ray_experiment(e, args)


if __name__ == "__main__":
    main()
