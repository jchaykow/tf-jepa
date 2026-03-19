import numpy as np
import torch


def DataTransform(sample, config):

    #weak_aug = scaling(sample, config.augmentation.jitter_scale_ratio)
    weak_aug = sample
    #strong_aug = jitter(permutation(sample, max_segments=config.augmentation.max_seg), config.augmentation.jitter_ratio)
    strong_aug = jitter(sample, config.window_len, config.augmentation.jitter_ratio)
    return weak_aug, strong_aug

def jitter(x: torch.Tensor, window_len: int, sigma: float = 0.8, noise_mag: float = 0.2):
    """Add Gaussian noise to a random contiguous sub-segment of every sample.

    The original implementation used nested Python loops and NumPy which
    became a major bottleneck for large batches. This vectorised version
    performs the same operation entirely in PyTorch and avoids any loops
    or excessive memory allocations.

    Args:
        x (torch.Tensor): Input tensor of shape (B, C, L).
        window_len (int): Total length *L* of the time-series window.
        sigma (float): Standard deviation multiplier for the Gaussian noise.
        noise_mag (float): Global scaling applied to the sampled noise.

    Returns:
        torch.Tensor: Tensor with noise injected, same shape as *x*.
    """

    # dimensions
    B, C, L = x.shape
    sub_len = int(0.7 * window_len)

    # guard against pathological configs
    if sub_len <= 0 or sub_len >= L:
        return x

    # choose a random starting index for every sample ‑- vectorised
    last_time_idx = window_len - sub_len
    device = x.device
    begin = torch.randint(0, last_time_idx, (B, 1, 1), device=device)

    # sample the noise once for every sample/channel/time-step we will modify
    # ensure the noise has the same dtype as the input tensor
    noise = (noise_mag * torch.randn(B, C, sub_len, device=device) * sigma).to(x.dtype)

    # build indices (broadcasted) and scatter-add the noise into a clone
    idx = begin + torch.arange(sub_len, device=device).view(1, 1, -1)  # (B,1,sub_len)
    xtilde = x.clone()
    xtilde.scatter_add_(2, idx.expand(-1, C, -1), noise)
    return xtilde
#     return x + np.random.normal(loc=0., scale=sigma, size=x.shape)ç


def scaling(x, sigma=1.1):
    # https://arxiv.org/pdf/1706.00527.pdf
    factor = np.random.normal(loc=2., scale=sigma, size=(x.shape[0], x.shape[2]))
    ai = []
    for i in range(x.shape[1]):
        xi = x[:, i, :]
        ai.append(np.multiply(xi, factor[:, :])[:, np.newaxis, :])
    return np.concatenate((ai), axis=1)


def permutation(x, max_segments=5, seg_mode="random"):
    orig_steps = np.arange(x.shape[2])

    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[2] - 2, num_segs[i] - 1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            ret[i] = pat[0,warp]
        else:
            ret[i] = pat
    return torch.from_numpy(ret)

