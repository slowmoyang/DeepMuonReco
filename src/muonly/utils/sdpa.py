from contextlib import nullcontext

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel


def sdpa_kernel_context(device: torch.device, backend: str):
    """Select configured scaled-dot-product-attention backend."""
    if backend == "default":
        return nullcontext()
    if backend == "math":
        if device.type == "cuda":
            return sdpa_kernel(SDPBackend.MATH)
        return nullcontext()
    raise ValueError(
        f"Unsupported SDPA backend: {backend}. Expected one of: default, math."
    )
