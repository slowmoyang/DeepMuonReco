from typing import Literal
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F


__all__ = [
    'RandomMasking',
]

class RandomMasking(nn.Module):

    """
    adapted from ...
    """

    def __init__(
        self,
        masking_ratio: float,
    ) -> None:
        """
        """
        super().__init__()
        self.masking_ratio = masking_ratio

    def forward(
        self,
        x: Tensor
    ) -> tuple[Tensor, Tensor]:
        """
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - self.masking_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = noise.argsort(dim=1) # ascend: small is keep, large is remove
        ids_restore = ids_shuffle.argsort(dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        ids_keep = ids_keep.unsqueeze(-1).repeat(1, 1, D)
        x_masked = x.gather(dim=1, index=ids_keep)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = x.new_ones(size=(N, L))
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = mask.gather(dim=1, index=ids_restore)

        return x_masked, mask


class MaskedMSELoss(nn.Module):


    def __init__(
        self,
        reduction: Literal['mean', 'sum', 'none'] = 'mean',
    ) -> None:
        super().__init__()

        self.reduction = reduction


    def forward(
        self,
        input: Tensor,
        target: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """
        Args:
            input: reconstructed data (N, L, D)
            target: actual data (N, L, D)
            mask: shape is(N, L). 0 (1) if visible (invisible)
        Returns:
            loss
        """
        loss = F.mse_loss(
            input=input,
            target=target,
            reduction='none',
        )

        visible_mask = mask.logical_not().unsqueeze(dim=2)

        loss = loss.masked_fill(
            mask=visible_mask,
            value=0,
        )

        if self.reduction == 'none':
            ...
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.sum() / mask.to(input).sum(dim=1)
        else:
            raise RuntimeError(f'{self.reduction=}')

        return loss
