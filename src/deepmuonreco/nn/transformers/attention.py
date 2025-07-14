r"""
- https://github.com/google-deepmind/hierarchical_perceiver/blob/b3074a4/perceiver_helpers.py
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import einops as eo


__all__ = [
    'CrossAttention',
    'SelfAttention',
]


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Tensor | None = None,
    dropout_p: float = 0.0,
) -> tuple[Tensor, Tensor]:
    """
    adapted from https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
    """
    scale_factor = 1 / math.sqrt(query.size(-1))

    if attn_mask is not None:
        N, H, T, _ = query.shape
        S = key.size(-2)
        attn_bias = query.new_zeros(N, H, T, S)
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias
    else:
        attn_bias = None

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    if attn_bias is not None:
        attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    output = attn_weight @ value
    return output, attn_weight


class CrossAttention(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        target_dim: int | None = None,
        source_dim: int | None = None,
        output_dim: int | None = None,
        dropout_p: float = 0,
        bias: bool = True,
    ) -> None:
        """
        """
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                f'embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})'
            )

        target_dim = target_dim or embed_dim
        source_dim = source_dim or embed_dim
        output_dim = output_dim or embed_dim

        self.num_heads = num_heads
        self._dropout_p = dropout_p

        self.query_projection = nn.Linear(
            in_features=target_dim,
            out_features=embed_dim,
            bias=bias,
        )
        self.key_projection = nn.Linear(
            in_features=source_dim,
            out_features=embed_dim,
            bias=bias,
        )
        self.value_projection = nn.Linear(
            in_features=source_dim,
            out_features=embed_dim,
            bias=bias,
        )
        self.output_projection = nn.Linear(
            in_features=embed_dim,
            out_features=output_dim,
            bias=bias,
        )
        self.output_dropout = nn.Dropout(
            p=dropout_p,
        )


    @property
    def dropout_p(self):
        return self._dropout_p if self.training else 0


    def forward(
        self,
        target: Tensor,
        source: Tensor,
        attn_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """
        """
        query = self.query_projection(target)
        key = self.key_projection(source)
        value = self.value_projection(source)

        query, key, value = [
            eo.rearrange(each, 'n l (h d) -> n h l d', h=self.num_heads)
            for each in [query, key, value]
        ]

        if attn_mask is not None:
            attn_mask = eo.repeat(
                tensor=attn_mask,
                pattern='n t s -> n h t s',
                h=self.num_heads,
            )

        output, attn_weight = scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p,
        )

        output = eo.rearrange(
            tensor=output,
            pattern='n h t d -> n t (h d)',
        )

        output = self.output_projection(output)
        output = self.output_dropout(output)
        return output, attn_weight


class SelfAttention(CrossAttention):

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        input_dim: int | None = None,
        output_dim: int | None = None,
        dropout_p: float = 0,
        bias: bool = True
    ) -> None:
        """
        """
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            target_dim=input_dim,
            source_dim=input_dim,
            output_dim=output_dim,
            dropout_p=dropout_p,
            bias=bias,
        )

    def forward( # type: ignore[override]
        self,
        input: Tensor,
        attn_mask: Tensor | None
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            input:
            attn_mask:
        Returns:
            output
        """
        return super().forward(
            target=input,
            source=input,
            attn_mask=attn_mask
        )
