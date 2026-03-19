import json, pickle, torch, numpy as np
from pathlib import Path
from scipy import signal  # pip install scipy

# ───────── config ─────────
DS_NAME      = "Epilepsy"                         # will be the --ds_name
TRAIN_PT     = "./data/Epilepsy/train.pt"
TEST_PT      = "./data/Epilepsy/test.pt"          # ‼️ already provided
RATE_IN      = 174                                # Hz in .pt blobs
RATE_OUT     = 64                                 # NormWear’s default CWT
OUT_ROOT     = Path(f"./data/{DS_NAME}")
PICKLE_DIR   = OUT_ROOT / "sample_for_downstream"
PICKLE_DIR.mkdir(parents=True, exist_ok=True)

def downsample(arr, r_in=RATE_IN, r_out=RATE_OUT):
    if r_in == r_out:
        return arr
    gcd = np.gcd(r_in, r_out)
    return signal.resample_poly(arr, r_out // gcd, r_in // gcd, axis=-1)

def build_split(pt_path, tag, uid_start=0):
    blob   = torch.load(pt_path, map_location="cpu")
    x_all  = blob["samples"].float().numpy()      # (N, 1, 178)
    y_all  = blob["labels"].int().numpy()         # (N,)
    fnames = []

    for i, (x, y) in enumerate(zip(x_all, y_all), uid_start):
        uid = f"{tag}-{i:05d}"
        x64 = downsample(x, RATE_IN, RATE_OUT).astype(np.float16)  # (1, T')
        fname = f"{uid}.pkl"
        pickle.dump(
            {
                "uid": uid,
                "data": x64,                      # [1, T']
                "sampling_rate": RATE_OUT,
                "label": [{"class": int(y)}],     # single-class label
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
