# convert_sleepeeg_normwear.py
import json, pickle, torch, numpy as np
from pathlib import Path

# ───────── config ─────────
DS_NAME   = "SleepEEG"
TRAIN_PT  = "./data/SleepEEG/train.pt"   # adjust if your path differs
TEST_PT   = "./data/SleepEEG/test.pt"
RATE      = 100                          # Hz already
OUT_ROOT  = Path(f"./data/{DS_NAME}")
PICKLE_DIR = OUT_ROOT / "sample_for_downstream"
PICKLE_DIR.mkdir(parents=True, exist_ok=True)

def build_split(pt_path: str, tag: str, uid_start: int = 0):
    blob  = torch.load(pt_path, map_location="cpu")
    xs    = blob["samples"].float().numpy()         # (N, 1, 200)
    fnames = []

    for i, x in enumerate(xs, uid_start):
        uid   = f"{tag}-{i:05d}"
        fname = f"{uid}.pkl"
        pickle.dump(
            {
                "uid": uid,
                "data": x.astype(np.float16),   # [1, 200]
                "sampling_rate": RATE,
                "label": [{"class": -1}],       # dummy label; ignored in SSL
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
      len(split['train']), "train samples and",
      len(split['test']),  "test samples written.")
