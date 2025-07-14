import torch
import torch.nn as nn
from torch import Tensor
from .. import TransformerDecoder


class InnerTrackSelectionTransformer(nn.Module):

    def __init__(
        self,
        track_dim: int,
        segment_dim: int,
        rechit_dim: int,
        model_dim: int,
        num_heads: int,
        num_layers: int,
        output_dim: int = 1,
        dropout_p: float = 0,
        widening_factor: int = 4,
    ) -> None:
        super().__init__()

        self.num_heads = num_heads

        # inner track embedding (px, py, eta)
        self.track_embedder = nn.Linear(
            in_features=track_dim,
            out_features=model_dim,
        )

        # DT/CSC segment embedding (position + direction; dimension: 6)
        self.segment_embedder = nn.Linear(
            in_features=segment_dim,
            out_features=model_dim,
        )

        # RPC/GEM rechit embedding (position only; dimension: 3)
        self.rechit_embedder = nn.Linear(
            in_features=rechit_dim,
            out_features=model_dim,
        )

        self.encoder = TransformerDecoder(
            embed_dim=model_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            widening_factor=widening_factor,
            dropout_p=dropout_p,
            self_attn=False, # no self-attention for target array (=track)
        )

        self.classification_head = nn.Linear(
            in_features=model_dim,
            out_features=output_dim,
        )

    def forward(
        self,
        track: Tensor,
        track_data_mask: Tensor,
        segment: Tensor,
        segment_data_mask: Tensor,
        rechit: Tensor,
        rechit_data_mask: Tensor,
    ) -> Tensor:
        """
        Args:
            track: (N, L_trk, D_trk)
            track_data_mask: (N, L_trk)
            segment: (N, L_seg, D_seg)
            segment_data_mask: (N, L_seg)
            rechit: (N, L_rec, D_rechit)
            rechit_data_mask: (N, L_rec)

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
        source_embed = torch.cat([segment_embed, rechit_embed], dim=1)
        # memory data_mask: (N, L_seg + L_rec)
        source_data_mask = torch.cat([segment_data_mask, rechit_data_mask], dim=1)

        # Transformer decoder: track_embed attends to memory_embed (DT/CSC segment + RPC/GEM rechit)
        latent = self.encoder(
            target=track_embed,
            source=source_embed,
            target_data_mask=track_data_mask,
            source_data_mask=source_data_mask,
        )

        # classification head: (N, L_trk, D_model) -> (N, L_trk, 1)
        logits = self.classification_head(latent)
        # squeeze: (N, L_trk, 1) -> (N, L_trk)
        logits = logits.squeeze(dim=2)
        logits = logits.masked_fill(mask=track_data_mask, value=0)
        return logits
