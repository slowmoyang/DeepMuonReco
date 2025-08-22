from torch import Tensor
import torch.nn as nn


__all__ = [
    'MaskedSelect',
    'MaskedMean',
]

# class MaskedSelect(nn.Module):
#
#     def forward(self, mask: Tensor, *args: Tensor, **kwargs: Tensor
#     ) -> tuple[tuple[Tensor, ...], dict[str, Tensor]]:
#         if mask.dtype is not torch.bool:
#             raise TypeError('Mask must be a boolean tensor')
#
#         if (len(args) + len(kwargs)) == 0:
#             raise ValueError('At least one argument is required')
#
#         args = tuple(each.masked_select(mask) for each in args)
#         kwargs = {key: value.masked_select(mask) for key, value in kwargs.items()}
#         return args, kwargs


class MaskedSelect(nn.Module):

    def forward(self, mask: Tensor, input: Tensor) -> Tensor:
        return input.masked_select(mask)


class MaskedMean(nn.Module):

    def forward(self, mask: Tensor, input: Tensor) -> Tensor:
        return input.masked_select(mask).mean()
