"""Device selection utilities."""

from __future__ import annotations

import torch


def get_device(preferred: str = "cuda") -> torch.device:
    """Return the best available torch device.

    Args:
        preferred: Preferred device string ('cuda' or 'cpu').

    Returns:
        Resolved torch.device with CUDA fallback.
    """
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
