"""Single source of truth for random-state control.

Reproducibility contract: every entry point calls ``set_seed(cfg.seed)``
exactly once, before any data shuffling or model construction.
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    """Seed every RNG the project uses.

    Torch is imported lazily: the data/feature/GBM stages must not
    require it (it is only installed from Phase 10 onward).
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
