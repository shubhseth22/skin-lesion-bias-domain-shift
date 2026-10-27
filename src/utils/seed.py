"""
Reproducibility and Hardware Device Utilities

Provides centralized seed management across PyTorch, NumPy, and standard library
random generators, and handles hardware device selection (CUDA, MPS, CPU).
"""

import os
import random
from typing import Optional
import numpy as np


def seed_everything(seed: int = 42, deterministic: bool = True) -> int:
    """
    Sets deterministic seeds across Python standard library, NumPy, and PyTorch (if available).

    Args:
        seed: Integer seed value.
        deterministic: If True, sets CuDNN to deterministic mode (may incur performance overhead).

    Returns:
        The integer seed value set.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    return seed


def get_device(preferred_device: Optional[str] = None) -> str:
    """
    Determines optimal compute device based on hardware availability and user preference.

    Checks for CUDA (NVIDIA GPUs), MPS (Apple Silicon), and falls back to CPU.

    Args:
        preferred_device: Optional override string ('cuda', 'mps', 'cpu').

    Returns:
        String identifier of selected device.
    """
    if preferred_device is not None:
        return preferred_device

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass

    return "cpu"
