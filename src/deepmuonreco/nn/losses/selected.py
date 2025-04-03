from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class SelectedBCEWithLogitsLoss(nn.Module):

    pos_weight: Tensor | None

    def __init__(self, pos_weight: Tensor | None) -> None:
        super().__init__()

        self.register_buffer('pos_weight', pos_weight)

    def forward(
        self,
        input: Tensor,
        target: Tensor,
        data_mask: Tensor,
    ) -> Tensor:
        input = input.masked_select(data_mask)
        target = target.masked_select(data_mask)
        loss = F.binary_cross_entropy_with_logits(input, target, reduction='none', pos_weight=self.pos_weight)
        return loss
