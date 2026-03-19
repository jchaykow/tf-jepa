# src/distributed/ray_datasets.py
"""
Convert the *.pt* splits you ship with each dataset into Ray AIR `Dataset`s
so they can be fed to TorchTrainer.  Works for any dataset folder that
follows your <train.pt | test.pt> convention.

Dependencies: only Ray and the utilities already in src/utils.py
"""

from __future__ import annotations
import os, pathlib
from typing import Tuple, Dict

import numpy as np
import torch
import torch.fft as fft
import ray

# ──────────────────────────────────────────────────────────────
# Reuse your existing augmentations
# ──────────────────────────────────────────────────────────────
from utils import (
    DataTransform_TD,   # time-domain jitter / scaling etc.
    DataTransform_FD,   # frequency-domain masking etc.
)

# ──────────────────────────────────────────────────────────────
# 1 ─ helper to load the raw *.pt split
# ──────────────────────────────────────────────────────────────
def _load_pt(file_path: str) -> Dict[str, torch.Tensor] | None:
    """Return dict from torch.load or None if the file is missing."""
    return (torch.load(file_path, weights_only=True)
            if os.path.exists(file_path) else None)


# ──────────────────────────────────────────────────────────────
# 2 ─ convert tensors → list[dict] compatible with Ray Dataset
# ──────────────────────────────────────────────────────────────
def _tensor_to_items(samples: torch.Tensor,
                     labels: torch.Tensor,
                     cfg: Dict,
                     training_mode: str) -> list[dict]:

    if samples.ndim == 2:                          # (N, T) -> (N, 1, T)
        samples = samples.unsqueeze(1)
    if samples.shape.index(min(samples.shape)) != 1:
        samples = samples.permute(0, 2, 1)

    samples = samples[:, :1, : cfg["TSlength_aligned"]]
    labels  = labels.long()

    samples_f = fft.fft(samples).abs()

    if training_mode == "pre_train":
        aug_t = DataTransform_TD(samples, cfg)
        aug_f = DataTransform_FD(samples_f, cfg)
    else:
        aug_t, aug_f = samples, samples_f

    items = []
    for i in range(samples.size(0)):
        items.append({
            "x_data":   samples[i].cpu().numpy(),
            "y_data":   int(labels[i]),
            "x_data_f": samples_f[i].cpu().numpy(),
            "aug_t":    aug_t[i].cpu().numpy(),
            "aug_f":    aug_f[i].cpu().numpy(),
        })
    return items


def _make_split(dataset_dir: str,
                split: str,
                cfg: Dict,
                training_mode: str,
                write: bool) -> ray.data.Dataset | None:
    """
    Build Ray Dataset from <dataset_dir>/<split>.pt.
    If file is missing return None.
    """
    pt_file = os.path.join(dataset_dir, f"{split}.pt")
    raw = _load_pt(pt_file)
    if raw is None:
        return None

    ds = ray.data.from_items(
        _tensor_to_items(raw["samples"], raw["labels"], cfg, training_mode)
    )
    if write:
        ds.write_parquet(pathlib.Path(dataset_dir) / f"{split}_ds.parquet")
    return ds

# ──────────────────────────────────────────────────────────────
# 3 ─ public API
# ──────────────────────────────────────────────────────────────
def make_ray_datasets(
    dataset_dir: str,
    cfg_dict: Dict,
    training_mode: str = "pre_train",
    write_parquet: bool = False,
) -> Tuple[ray.data.Dataset | None,
           ray.data.Dataset | None,
           ray.data.Dataset | None]:
    """
    Create Ray datasets for train / val / test splits that actually exist
    in `dataset_dir`.
    """
    train_ds = _make_split(dataset_dir, "train", cfg_dict,
                           training_mode, write_parquet)
    val_ds   = _make_split(dataset_dir, "val",   cfg_dict,
                           training_mode, write_parquet)
    test_ds  = _make_split(dataset_dir, "test",  cfg_dict,
                           training_mode, write_parquet)

    return train_ds, val_ds, test_ds


# convenience loader that main_ray.py already uses
def load_parquet_shard(dataset_dir: str, split: str) -> ray.data.Dataset:
    path = pathlib.Path(dataset_dir) / f"{split}_ds.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found.  Run `export_to_ray_datasets.py` first."
        )
    return ray.data.read_parquet(path)
