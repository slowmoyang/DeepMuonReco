import torch
import torch.nn as nn
from torch import Tensor
import einops as eo


def make_cross_attn_mask(
    source_pad_mask: Tensor,
    target_pad_mask: Tensor,
    num_heads: int,
) -> Tensor:
    target_len = target_pad_mask.size(1)

    attn_mask = eo.repeat(
        tensor=source_pad_mask,
        pattern='n s -> (n h) t s',
        h=num_heads,
        t=target_len,
    )
    return attn_mask


def make_self_attn_mask(
    pad_mask: Tensor,
    num_heads: int,
) -> Tensor:
    return make_cross_attn_mask(
        source_pad_mask=pad_mask,
        target_pad_mask=pad_mask,
        num_heads=num_heads
    )


@torch.no_grad()
def init_params(module: nn.Module) -> None:
    """
    """
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
