import torch
import torch.nn as nn
from torch import Tensor
from ..utils import make_cross_attn_mask, make_self_attn_mask
from deepmuonreco.data import InnerTrackSelectionDataset


class InnerTrackSelectionTransformer(nn.Module):

    def __init__(
        self,
        dim_track: int = InnerTrackSelectionDataset.DIM_TRACK,
        dim_segment: int = InnerTrackSelectionDataset.DIM_SEGMENT,
        dim_rechit: int = InnerTrackSelectionDataset.DIM_RECHIT,
        dim_output: int = InnerTrackSelectionDataset.DIM_TARGET,
        dim_model: int = 64,
        dim_feedforward: int = 128,
        activation: str = 'relu',
        num_heads: int = 2,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.num_heads = num_heads

        # Track embedding (px, py, eta)
        self.track_embedder = nn.Linear(
            in_features=dim_track,
            out_features=dim_model,
        )

        # DT/CSC segment embedding (position + direction; dimension: 6)
        self.segment_embedder = nn.Linear(
            in_features=dim_segment,
            out_features=dim_model,
        )

        # RPC/GEM rechit embedding (position only; dimension: 3)
        self.rechit_embedder = nn.Linear(
            in_features=dim_rechit,
            out_features=dim_model,
        )

        layer = nn.TransformerDecoderLayer(
            d_model=dim_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=False,
            bias=True,
        )

        self.backbone = nn.TransformerDecoder(
            decoder_layer=layer,
            num_layers=num_layers,
        )

        self.classification_head = nn.Linear(
            in_features=dim_model,
            out_features=dim_output,
        )

    def forward(
        self,
        track: Tensor,
        track_pad_mask: Tensor,
        segment: Tensor,
        segment_pad_mask: Tensor,
        rechit: Tensor,
        rechit_pad_mask: Tensor,
    ) -> Tensor:
        """
        Args:
            track: (N, L_trk, D_trk)
            track_pad_mask: (N, L_trk)
            segment: (N, L_seg, D_seg)
            segment_pad_mask: (N, L_seg)
            rechit: (N, L_rec, D_rechit)
            rechit_pad_mask: (N, L_rec)

        Returns:
            logits: (N, L_trk)
        """
        # embed track features: (N, L_trk, D_model)
        track_embed = self.track_embedder(track)
        # embed segment features (DT/CSC): (N, L_seg, D_model)
        segment_embed = self.segment_embedder(segment)
        # embed rechit features (RPC/GEM): (N, L_rec, D_model)
        rechit_embed = self.rechit_embedder(rechit)
        # concatenate segment and rechit dimension -> memory tensor
        # embed: (N, L_seg + L_rec, D_model)
        memory_embed = torch.cat([segment_embed, rechit_embed], dim=1)
        # memory pad_mask: (N, L_seg + L_rec)
        memory_pad_mask = torch.cat([segment_pad_mask, rechit_pad_mask], dim=1)

        # compute self-attention mask for track features (target)
        tgt_mask = make_self_attn_mask(
            pad_mask=track_pad_mask,
            num_heads=self.num_heads,
        )

        # compute cross-attention mask between track (target) and combined memory
        memory_mask = make_cross_attn_mask(
            source_pad_mask=memory_pad_mask,
            target_pad_mask=track_pad_mask,
            num_heads=self.num_heads,
        )

        # Transformer decoder: track_embed attends to memory_embed (DT/CSC segment + RPC/GEM rechit)
        latent = self.backbone(
            tgt=track_embed,
            memory=memory_embed,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            memory_key_padding_mask=memory_pad_mask,
            tgt_key_padding_mask=track_pad_mask,
            tgt_is_causal=False,
            memory_is_causal=False,
        )

        # classification head: (N, L_trk, D_model) -> (N, L_trk, 1)
        logits: Tensor = self.classification_head(latent)
        # squeeze: (N, L_trk, 1) -> (N, L_trk)
        logits = logits.squeeze(dim=2)
        logits = logits.masked_fill(mask=track_pad_mask, value=0)
        return logits