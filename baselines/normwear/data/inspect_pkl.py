# inspect_pkl.py
import pickle, torch, sys, numpy as np
from pathlib import Path

pkl_path = Path(sys.argv[1])           # first CLI arg

with open(pkl_path, "rb") as f:
    blob = pickle.load(f)              # or torch.load(..., map_location="cpu")

print("Top-level type:", type(blob))

for k, v in blob.items():
    if isinstance(v, np.ndarray):
        print(f"{k}: ndarray, shape={v.shape}, dtype={v.dtype}")
    elif torch.is_tensor(v):
        print(f"{k}: tensor, shape={tuple(v.shape)}, dtype={v.dtype}")
    else:
        print(f"{k}: {type(v)} → {v}")
