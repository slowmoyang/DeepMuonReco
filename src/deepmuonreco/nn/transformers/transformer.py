"""
pre-norm modules
"""
from torch import Tensor
import torch.nn as nn
from torch.nn.modules.transformer import _get_clones
import einops as eo
from .attention import CrossAttention
from .attention import SelfAttention
from .mlp import MultiLayerPerceptron


__all__ = [
    'CrossAttentionBlock',
    'SelfAttentionBlock',
    'MLPBlock',
    'TransformerEncoder',
    'TransformerDecoder',
]


class CrossAttentionBlock(nn.Module):
    """Cross-attention sub-block
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        use_post_attention_residual: bool,
        target_dim: int | None = None,
        source_dim: int | None = None,
        output_dim: int | None = None,
        dropout_p: float = 0,
        bias: bool = True,
    ) -> None:
        """
        """
        super().__init__()

        target_dim = target_dim or embed_dim
        source_dim = source_dim or embed_dim

        self.use_post_attention_residual = use_post_attention_residual

        self.norm_src = nn.LayerNorm(
            normalized_shape=source_dim,
            bias=bias,
        )
        self.norm_tgt = nn.LayerNorm(
            normalized_shape=target_dim,
            bias=bias,
        )
        self.attn = CrossAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            target_dim=target_dim,
            source_dim=source_dim,
            output_dim=output_dim,
            dropout_p=dropout_p,
            bias=bias,
        )

    def forward(
        self,
        target: Tensor,
        source: Tensor,
        attn_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """
        """
        # identity
        identity = target

        # residual function, norm first
        target = self.norm_tgt(target)
        source = self.norm_src(source)

        output, attn_weight = self.attn(
            target=target,
            source=source,
            attn_mask=attn_mask
        )

        if self.use_post_attention_residual:
            output = identity + output
        return output, attn_weight


class SelfAttentionBlock(nn.Module):

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
        super().__init__()
        input_dim = input_dim or embed_dim

        self.norm = nn.LayerNorm(
            normalized_shape=input_dim,
            bias=bias,
        )
        self.attn = SelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            input_dim=input_dim,
            output_dim=output_dim,
            dropout_p=dropout_p,
            bias=bias,
        )

    def forward(
        self,
        input: Tensor,
        attn_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            input: tensor, float, (N, L, E)
        Returns:
        """
        # identity
        identity = input

        # residual function
        output = self.norm(input)
        output, attn_weight = self.attn(
            input=output,
            attn_mask=attn_mask,
        )

        # skip connection
        output = identity + output
        return output, attn_weight


class MLPBlock(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        widening_factor: int,
        bias: bool = True,
        dropout_p: float = 0,
    ) -> None:
        """
        """
        super().__init__()

        self.norm = nn.LayerNorm(
            normalized_shape=embed_dim,
            bias=bias,
        )
        self.mlp = MultiLayerPerceptron(
            input_dim=embed_dim,
            output_dim=None,
            widening_factor=widening_factor,
            bias=bias,
            dropout_p=dropout_p,
        )


    def forward(
        self,
        input: Tensor,
    ) -> Tensor:
        """
        """
        # identity
        identity = input
        # residual
        output = self.norm(input)
        output = self.mlp(output)
        # skip connection
        output = identity + output
        return output



class TransformerEncoderLayer(nn.Module):


    def __init__(
        self,
        latent_dim: int,
        num_heads: int,
        widening_factor: int,
        dropout_p: float = 0,
    ) -> None:
        """
        """
        super().__init__()
        self.attention = SelfAttentionBlock(
            embed_dim=latent_dim,
            num_heads=num_heads,
            dropout_p=dropout_p,
        )
        self.mlp = MLPBlock(
            embed_dim=latent_dim,
            widening_factor=widening_factor,
            dropout_p=dropout_p,
        )

    def forward(
        self,
        input: Tensor,
        attn_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """
        """
        output, attn_weight = self.attention(
            input=input,
            attn_mask=attn_mask,
        )
        output = self.mlp(
            input=output,
        )
        return output, attn_weight



class TransformerEncoder(nn.Module):
    """Transformer encoder
    """

    def __init__(
        self,
        latent_dim: int,
        num_heads: int,
        num_layers: int = 1,
        widening_factor: int = 1,
        dropout_p: float = 0,
    ) -> None:
        """
        """
        super().__init__()

        layer = TransformerEncoderLayer(
            latent_dim=latent_dim,
            num_heads=num_heads,
            widening_factor=widening_factor,
            dropout_p=dropout_p,
        )

        self.layer_list = nn.ModuleList(_get_clones(layer, num_layers))

    def forward(
        self,
        input: Tensor,
        data_mask: Tensor | None,
    ) -> Tensor:
        """
        """
        if data_mask is None:
            attn_mask = None
        else:
            # n: batch size, s: source array length, t: target array length
            attn_mask = eo.repeat(
                tensor=data_mask,
                pattern='n s -> n t s',
                t=data_mask.size(1),
            )

        output = input
        for layer in self.layer_list:
            output, _ = layer(input=output, attn_mask=attn_mask)

        if data_mask is not None:
            output.masked_fill_(
                mask=data_mask.unsqueeze(-1).logical_not(),
                value=0,
            )
        return output


class TransformerDecoderLayer(nn.Module):


    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        widening_factor: int,
        dropout_p: float = 0,
        self_attn: bool = True
    ) -> None:
        """
        """
        super().__init__()
        if self_attn:
            self.self_attn = SelfAttentionBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout_p=dropout_p,
            )
        else:
            self.self_attn = None

        self.cross_attn = CrossAttentionBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout_p=dropout_p,
            use_post_attention_residual=False,
        )
        self.mlp = MLPBlock(
            embed_dim=embed_dim,
            widening_factor=widening_factor,
            dropout_p=dropout_p,
        )

    def forward(
        self,
        target: Tensor,
        source: Tensor,
        self_attn_mask: Tensor | None,
        cross_attn_mask: Tensor | None,
    ) -> Tensor:
        """
        """
        output = target
        if self.self_attn is not None:
            output, _ = self.self_attn(
                input=output,
                attn_mask=self_attn_mask,
            )

        output, _ = self.cross_attn(
            target=output,
            source=source,
            attn_mask=cross_attn_mask,
        )

        output = self.mlp(
            input=output,
        )
        return output



class TransformerDecoder(nn.Module):
    """Transformer encoder
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_layers: int = 1,
        widening_factor: int = 1,
        dropout_p: float = 0,
        self_attn: bool = True,
    ) -> None:
        """
        """
        super().__init__()

        self._do_self_attn = self_attn

        layer = TransformerDecoderLayer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            widening_factor=widening_factor,
            dropout_p=dropout_p,
            self_attn=self_attn,
        )

        self.layer_list = nn.ModuleList(_get_clones(layer, num_layers))

    def _make_self_attn_mask(self, target_data_mask: Tensor | None) -> Tensor | None:
        if not self._do_self_attn:
            return None

        if target_data_mask is None:
            self_attn_mask = None
        else:
            # n: batch size, s: source array length, t: target array length
            self_attn_mask = eo.repeat(
                tensor=target_data_mask,
                pattern='n s -> n t s',
                t=target_data_mask.size(1),
            )
        return self_attn_mask

    def _make_cross_attn_mask(self, source_data_mask: Tensor | None, target: Tensor) -> Tensor | None:
        if source_data_mask is None:
            cross_attn_mask = None
        else:
            # n: batch size, s: source array length, t: target array length
            cross_attn_mask = eo.repeat(
                tensor=source_data_mask,
                pattern='n s -> n t s',
                t=target.size(1), # FIXME:
            )
        return cross_attn_mask

    def forward(
        self,
        target: Tensor,
        source: Tensor,
        target_data_mask: Tensor | None = None,
        source_data_mask: Tensor | None = None,
    ) -> Tensor:
        """
        """
        self_attn_mask = self._make_self_attn_mask(target_data_mask)
        cross_attn_mask = self._make_cross_attn_mask(source_data_mask, target)

        output = target
        for layer in self.layer_list:
            output = layer(
                target=output,
                source=source,
                self_attn_mask=self_attn_mask,
                cross_attn_mask=cross_attn_mask,
            )

        if target_data_mask is not None:
            output.masked_fill_(
                mask=target_data_mask.unsqueeze(-1).logical_not(),
                value=0,
            )
        return output
