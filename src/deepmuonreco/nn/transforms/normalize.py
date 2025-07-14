import torch
from torch import nn, Tensor


__all__ = [
    'Normalize',
]


class Normalize(nn.Module):
    """torchvision-like Normalize"""

    mean: Tensor
    std: Tensor

    def __init__(
        self,
        mean,
        std,
    ) -> None:
        super().__init__()
        mean = torch.tensor(mean, dtype=torch.float32)
        std = torch.tensor(std, dtype=torch.float32)

        self.register_buffer('mean', mean)
        self.register_buffer('std', std)


    def forward(self, input: Tensor) -> Tensor:
        # Check if input is empty
        if input.numel() == 0:
            return input
        return (input - self.mean) / self.std

