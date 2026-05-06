import logging
import random
import numpy as np
import torch


_logger = logging.getLogger(__name__)

__all__ = [
    "set_seed",
]


def set_seed(seed: int) -> None:
    _logger.info(f"Setting global random seed to {seed}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
