"""
Convert FD train/test .pt blobs → NormWear pickle layout
-------------------------------------------------------

Input  structure  (inside each .pt):
    {
        "samples": Tensor[N, 1, 5120],   # raw waveform windows
        "labels" : Tensor[N]             # one class per window
    }

Output structure expected by NormWear:
    data/FD/
        ├─ sample_for_downstream/
        │    ├─ train-00000.pkl
        │    ├─ …
        │    └─ test-10000.pkl
        └─ train_test_split.json         # lists the above filenames
"""

import json, pickle, torch, numpy as np
from pathlib import Path
from scipy import signal           # pip install scipy

# ────────── config ──────────
DS_NAME   = "FD"
TRAIN_PT  = "./data/FD/train.pt"
TEST_PT   = "./data/FD/test.pt"
RATE_IN   = 64_000                 # 64 k Hz
RATE_OUT  = 256                    # choose 64 if you want parity with NormWear
OUT_ROOT  = Path(f"./data/{DS_NAME}")
PICKLE_DIR = OUT_ROOT / "sample_for_downstream"
PICKLE_DIR.mkdir(parents=True, exist_ok=True)

def downsample(arr, r_in=RATE_IN, r_out=RATE_OUT):
    """Down-sample a 1×L or C×L numpy array with polyphase filtering."""
    if r_in == r_out:
        return arr
    gcd = np.gcd(r_in, r_out)
    up, down = r_out // gcd, r_in // gcd
    return signal.resample_poly(arr, up, down, axis=-1)

def build_split(pt_path: str, tag: str, uid_start: int = 0):
    """Convert a single .pt file → many .pkl files. Return filename list."""
    blob   = torch.load(pt_path, map_location="cpu")
    x_all  = blob["samples"].float().numpy()  # (N, 1, 5120)
    y_all  = blob["labels"].int().numpy()     # (N,)
    fnames = []

    for i, (x, y) in enumerate(zip(x_all, y_all), uid_start):
        uid   = f"{tag}-{i:05d}"
        x_ds  = downsample(x, RATE_IN, RATE_OUT).astype(np.float16)  # (1, L')
        fname = f"{uid}.pkl"
        pickle.dump(
            {
                "uid": uid,
                "data": x_ds,                       # shape [1, L']
                "sampling_rate": RATE_OUT,
                "label": [{"class": int(y)}],
            },
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

print("✔  Finished —",
      len(split["train"]), "train samples and",
      len(split["test"]),  "test samples written.")
