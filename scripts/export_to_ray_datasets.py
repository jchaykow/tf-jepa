#!/usr/bin/env python3
"""
export_to_parquet.py
Turn every <dataset>/{train.pt, test.pt} into
<dataset>/{train_ds.parquet, finetune_ds.parquet, test_ds.parquet}.
Run it once; afterwards main_ray.py will just read the Parquet.
"""
import argparse, pathlib, json
from tqdm import tqdm
import ray

from distributed.ray_datasets import make_ray_datasets
from configs      import EXPERIMENTS, to_dict, load_config


def export_for_folder(folder: pathlib.Path, cfg_dict, overwrite: bool):
    need = [folder / f"{s}_ds.parquet" for s in ("train", "val", "test")]
    if not overwrite and all(p.exists() for p in need):
        print(f"✔ {folder.name}: Parquet already exists – skip")
        return
    print(f"↻  {folder.name}: generating Parquet …")
    make_ray_datasets(
        dataset_dir=str(folder),
        cfg_dict=cfg_dict,
        training_mode="pre_train",
        write_parquet=True,
    )
    print(f"✔ {folder.name}: done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="all",
                    help="Experiment key (har, ecg …) or 'all'")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    ray.init(ignore_reinit_error=True, num_cpus=1, include_dashboard=False)

    keys = (list(EXPERIMENTS) if args.exp.lower() == "all"
            else [args.exp.lower()])

    for k in keys:
        if k not in EXPERIMENTS:
            raise ValueError(f"Unknown key '{k}'. Allowed: {list(EXPERIMENTS)}")

        # one config per experiment is enough for both folders
        cfg_obj  = load_config(EXPERIMENTS[k]["config_module"])
        cfg_dict = to_dict(cfg_obj)

        src_dir = pathlib.Path(EXPERIMENTS[k]["source_data"])
        tgt_dir = pathlib.Path(EXPERIMENTS[k]["target_data"])

        export_for_folder(src_dir, cfg_dict, args.overwrite)
        export_for_folder(tgt_dir, cfg_dict, args.overwrite)


if __name__ == "__main__":
    main()
