import torch
from torch import nn, Tensor


__all__ = [
    'Normalize',
]


class Normalize(nn.Module):
    """torchvision-like Normalize"""

    offset: Tensor
    scale: Tensor

    def __init__(
        self,
        offset,
        scale,
    ) -> None:
        super().__init__()
        offset = torch.tensor(offset, dtype=torch.float32)
        scale = torch.tensor(scale, dtype=torch.float32)

        self.register_buffer('offset', offset)
        self.register_buffer('scale', scale)


    def forward(self, input: Tensor) -> Tensor:
        # Check if input is empty
        if input.numel() == 0:
            return input
        return (input - self.offset) / self.scale

