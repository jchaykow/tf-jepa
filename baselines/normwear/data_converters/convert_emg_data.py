"""
Convert EMG train/test .pt files  ➜  NormWear pickle layout
-----------------------------------------------------------

Expected input (each .pt):
    {
        "samples": Tensor[N, 1, 1500],   # 1-ch EMG windows
        "labels" : Tensor[N]             # {0,1,2}
    }

Output directory:  data/EMG/
"""

import json, pickle, torch, numpy as np
from pathlib import Path
from scipy import signal          # pip install scipy

# ── config ───────────────────────────────────────────────────────────
DS_NAME   = "EMG"
TRAIN_PT  = "./data/EMG/train.pt"
TEST_PT   = "./data/EMG/test.pt"
RATE_IN   = 4_000                 # 4 k Hz raw
RATE_OUT  = 256                   # down-sample to 256 Hz (keeps signal, saves RAM)
OUT_ROOT  = Path(f"./data/{DS_NAME}")
PICKLE_DIR = OUT_ROOT / "sample_for_downstream"
PICKLE_DIR.mkdir(parents=True, exist_ok=True)

def ds(arr, r_in=RATE_IN, r_out=RATE_OUT):
    if r_in == r_out:
        return arr
    g = np.gcd(r_in, r_out)
    return signal.resample_poly(arr, r_out // g, r_in // g, axis=-1)

def build_split(pt_path, tag, uid_start=0):
    blob  = torch.load(pt_path, map_location="cpu")
    xs, ys = blob["samples"].float().numpy(), blob["labels"].int().numpy()
    fnames = []

    for i, (x, y) in enumerate(zip(xs, ys), uid_start):
        uid = f"{tag}-{i:05d}"
        x_  = ds(x).astype(np.float16)              # (1, ≈96)
        fname = f"{uid}.pkl"
        pickle.dump(
            {"uid": uid,
             "data": x_,
             "sampling_rate": RATE_OUT,
             "label": [{"class": int(y)}]},
            open(PICKLE_DIR / fname, "wb"),
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        fnames.append(fname)
    return fnames

split = {
    "train": build_split(TRAIN_PT, "train"),
    "test":  build_split(TEST_PT,  "test", uid_start=10_000),
}

with open(OUT_ROOT / "train_test_split.json", "w") as f:
    json.dump(split, f, indent=2)

print("✔  Done —", len(split['train']), "train &", len(split['test']), "test samples.")
