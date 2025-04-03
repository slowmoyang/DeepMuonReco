from torch import Tensor
import torch.nn as nn


class MaskedSelect(nn.Module):

    def forward(self, input: Tensor, mask: Tensor) -> Tensor:
        return input.masked_select(mask)
