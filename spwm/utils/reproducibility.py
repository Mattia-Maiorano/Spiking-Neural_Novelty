"""
Reproducibility utilities for SPWM.
Ensures deterministic seeding across Python, NumPy, PyTorch, and accelerator backends.
"""

from __future__ import annotations
import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Sets global seeds for complete experimental reproducibility.
    Configures Python, NumPy, PyTorch CPU, CUDA, and Apple Silicon MPS.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
