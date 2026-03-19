import torch, pprint, numpy as np

blob = torch.load("./data/epilepsy/train.pt", map_location="cpu")

# 1. High-level overview
print("Top-level type:", type(blob))
if isinstance(blob, dict):
    print("Keys:", list(blob.keys()))

# 2. Look at shapes/dtypes for the parts you care about
for k, v in blob.items():
    if isinstance(v, (torch.Tensor, np.ndarray)):
        print(f"{k}: shape={tuple(v.shape)}, dtype={v.dtype}")
    else:
        print(f"{k}: {type(v)}")
