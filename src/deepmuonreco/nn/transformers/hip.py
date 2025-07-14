"""
- https://arxiv.org/abs/2202.10890
- https://github.com/google-deepmind/hierarchical_perceiver/tree/b3074a4
"""
from typing import Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.transformer import _get_clones
from einops.layers.torch import Rearrange
from .perceiver import PerceiverEncoder
from .perceiver import PerceiverProcessor


__all__ = [
    'HiPBlock',
    'HiPBlockSequential',
    'HiPEncoder',
    'HiPDecoder',
]


class HiPBlock(nn.Module):

    def __init__(
        self,
        input_dim: int,
        # common
        latent_len: int,
        latent_dim: int,
        num_groups: int,
        # NOTE:
        encoder_num_heads: int,
        encoder_widening_factor: int,
        # NOTE:
        num_processors: int,
        processor_num_heads: int,
        processor_widening_factor: int,
        dropout_p: float = 0,
    ) -> None:
        """
        """
        super().__init__()

        self.grouper = Rearrange(
            pattern='n (g k) c -> (n g) k c',
            g=num_groups,
        )

        self.mask_grouper = Rearrange(
            pattern='n (g k) -> (n g) k',
            g=num_groups,
        )

        self.ungrouper = Rearrange(
            pattern='(n g) k c -> n (g k) c',
            g=num_groups,
        )

        self.encoder = PerceiverEncoder(
            latent_len=latent_len,
            latent_dim=latent_dim,
            num_heads=encoder_num_heads,
            use_post_attention_residual=True,
            widening_factor=encoder_widening_factor,
            input_dim=input_dim,
            dropout_p=dropout_p,
        )

        processor = PerceiverProcessor(
            latent_dim=latent_dim,
            num_heads=processor_num_heads,
            widening_factor=processor_widening_factor,
            dropout_p=dropout_p,
        )
        self.processor_list = _get_clones(processor, num_processors)

        self.num_groups = num_groups


    def forward(
        self,
        input: Tensor,
        pre_attention_residual: Tensor | None,
        data_mask: Tensor | None,
    ) -> Tensor:
        """
        Args:
            input: a
            data_mask:
        Returns:
            output (torch.Tensor):
        """
        x = input
        N, L, D = input.shape
        G = self.num_groups

        # NOTE: padding
        pad_size = G - (L % G)

        if pad_size != 0:
            # pad: (left, right, top, bottom, front, back)
            x = F.pad(input=x, pad=[0, 0, 0, pad_size, 0, 0], mode='constant', value=0)

            if data_mask is None:
                data_mask = x.new_full(size=(N, L), fill_value=True, dtype=torch.bool)
            # pad: (left, right, top, bottom)
            data_mask = F.pad(data_mask, pad=[0, pad_size, 0, 0], mode='constant', value=False)

        # NOTE: grouping
        x = self.grouper(x)
        if pre_attention_residual is not None:
            pre_attention_residual = self.grouper(pre_attention_residual)
        if data_mask is not None:
            data_mask = self.mask_grouper(data_mask)

        # NOTE: encoder

        z = self.encoder(
            input=x,
            data_mask=data_mask,
            pre_attention_residual=pre_attention_residual,
        )

        for processor in self.processor_list:
            z = processor(z)

        # NOTE: ungrouping
        z = self.ungrouper(z)

        return z


class HiPBlockSequential(nn.Module):

    def __init__(
        self,
        input_dim: int,
        # NOTE: shared
        latent_len: list[int],
        latent_dim: list[int],
        num_groups: list[int],
        # NOTE: processors
        num_processors: list[int],
        processor_num_heads: list[int],
        processor_widening_factor: list[int] | None = None,
        # NOTE: encoder
        encoder_num_heads: list[int] | None = None,
        encoder_widening_factor: list[int] | None = None,
    ) -> None:
        super().__init__()

        num_blocks = len(latent_len)

        processor_widening_factor = processor_widening_factor or ([4] * num_blocks)
        encoder_num_heads = encoder_num_heads or ([1] * num_blocks)
        encoder_widening_factor = encoder_widening_factor or ([1] * num_blocks)

        input_dim_list = [input_dim] + latent_dim[:-1]

        kwargs_dict: dict[str, list[int]] = dict(
            input_dim=input_dim_list,
            latent_len=latent_len,
            latent_dim=latent_dim,
            num_groups=num_groups,
            encoder_num_heads=encoder_num_heads,
            encoder_widening_factor=encoder_widening_factor,
            num_processors=num_processors,
            processor_num_heads=processor_num_heads,
            processor_widening_factor=processor_widening_factor,
        )

        kwargs_list: list[dict[str, Any]] = [
            dict(zip(kwargs_dict, each))
            for each in zip(*kwargs_dict.values())
        ]

        # HiP's encoder
        self.block_list = nn.ModuleList([
            HiPBlock(**kwargs) for kwargs in kwargs_list
        ])




class HiPEncoder(HiPBlockSequential):
    """Encoder part of the HiP (full) without input and position embedders"""

    def __init__(
        self,
        input_dim: int,
        latent_len: list[int],
        latent_dim: list[int],
        num_groups: list[int],
        num_processors: list[int],
        processor_num_heads: list[int],
        processor_widening_factor: list[int] | None = None,
        encoder_num_heads: list[int] | None = None,
        encoder_widening_factor: list[int] | None = None,
        return_hidden: bool = True,
    ) -> None:
        """
        """
        super().__init__(
            input_dim,
            latent_len,
            latent_dim,
            num_groups,
            num_processors,
            processor_num_heads,
            processor_widening_factor,
            encoder_num_heads,
            encoder_widening_factor,
        )

        self.return_hidden = return_hidden

    @property
    def last_block_idx(self) -> int:
        return len(self.block_list) - 1

    def forward(
        self,
        input: Tensor,
        data_mask: Tensor | None = None,
    ) -> tuple[Tensor, ...]:
        """
        Args:
            input:
            data_mask:
        Returns:
        """
        x = input

        # encoder
        output = []
        for idx, block in enumerate(self.block_list):
            x = block(
                input=x,
                pre_attention_residual=None,
                data_mask=(data_mask if idx == 0 else None),
            )

            if idx == self.last_block_idx:
                output.append(x)
            elif self.return_hidden:
                output.append(x)

        return tuple(reversed(output))


class HiPDecoder(HiPBlockSequential):
    """HiP (full) without input and position embedders"""


    def forward(
        self,
        input: Tensor,
        *hidden: Tensor,
    ) -> Tensor:
        """
        Args:
            input:
            input_data_mask:
        Returns:
        """
        x = input
        for idx, block in enumerate(self.block_list):
            x = block(
                input=x,
                pre_attention_residual=hidden[idx],
                data_mask=None,
            )
        return x
