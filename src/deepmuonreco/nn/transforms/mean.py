import torch.nn as nn
from torch import Tensor


__all__ = [
    'Mean',
    'MaskedMean',
]


class Mean(nn.Module):

    def forward(self, input: Tensor) -> Tensor:
        return input.mean()


class MaskedMean(nn.Module):

    def forward(
        self,
        input: Tensor,
        data_mask: Tensor,
    ) -> Tensor:
        return input.masked_select(data_mask).mean()
