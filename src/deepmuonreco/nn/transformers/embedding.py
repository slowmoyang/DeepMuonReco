import math
import torch
from torch import nn
from torch import Tensor


__all__ = [
    'TrainablePositionEmbedder',
]


class TrainablePositionEmbedder(nn.Module):

    def __init__(
        self,
        input_dim: int,
        input_len: int,
    ) -> None:
        super().__init__()

        # FIXME:
        self.embed = nn.Parameter(
            data=torch.empty(1, input_len, input_dim),
        )

        self.reset_parameters()

    def forward(
        self,
        input: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        """
        batch_size = input.size(0)
        embed = self.embed.repeat(batch_size, 1, 1)
        output = input + embed
        return output, embed

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.embed, a=math.sqrt(5))
