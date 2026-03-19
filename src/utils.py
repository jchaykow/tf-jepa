import torch
import math, torch.nn.functional as F


# ──────────────────────────────────────────────
# Generic helpers
# ──────────────────────────────────────────────
def _rand_mask(shape, keep_prob: float, device):
    """
    Return a boolean mask with `keep_prob` fraction of 1-values.
    Works on any device.
    """
    return (torch.rand(shape, device=device) < keep_prob)


# ──────────────────────────────────────────────
# Augmentations
# ──────────────────────────────────────────────
def jitter(x, sigma=0.8):
    # torch.randn_like uses x.device automatically
    return x + torch.randn_like(x) * sigma


def DataTransform_TD(sample, config):
    return jitter(sample, config["augmentation"]["jitter_ratio"])


def remove_frequency(x, pertub_ratio=0.0):
    """Randomly *zero out* a fraction `pertub_ratio` of freq bins."""
    keep_prob = 1.0 - pertub_ratio          # keep the rest
    mask = _rand_mask(x.shape, keep_prob, x.device)
    return x * mask


def add_frequency(x, pertub_ratio=0.0):
    """Randomly *add noise* to a fraction `pertub_ratio` of freq bins."""
    add_prob = pertub_ratio
    mask = _rand_mask(x.shape, add_prob, x.device)

    max_amp = x.max()
    noise   = torch.rand_like(x) * (max_amp * 0.1)
    return x + mask * noise


def DataTransform_FD(sample, config):
    aug1 = remove_frequency(sample, pertub_ratio=0.1)
    aug2 = add_frequency   (sample, pertub_ratio=0.1)
    return aug1 + aug2


# ──────────────────────────────────────────────
# Patch helper (unchanged)
# ──────────────────────────────────────────────
def patchify(x, patch_size=178):
    """
    x: (B, 1, L) → (B, N, 178)
    """
    B, _, L = x.shape
    n_patches = math.ceil(L / patch_size)

    # Pad on the right so L % patch_size == 0
    pad = n_patches * patch_size - L
    if pad:
        x = F.pad(x, (0, pad))

    return x.view(B, n_patches, patch_size)
