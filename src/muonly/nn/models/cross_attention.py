import torch
import torch.nn as nn
from torch import Tensor
from ..transformers.transformer import TransformerEncoder, TransformerDecoder


__all__ = [
    "CrossAttentionModel",
]


class CrossAttentionModel(nn.Module):
    """Asymmetric cross-attention preselection model.

    The number of tracker tracks per event (~7000) is far larger than the
    number of muon-system measurements (~88). The muon side is therefore
    encoded with self-attention (cheap), and each tracker track is scored
    independently by cross-attending to that encoded muon memory. There is no
    track self-attention, so the cost is linear in the number of tracker
    tracks instead of quadratic.
    """

    def __init__(
        self,
        # input dimensions
        tracker_track_dim: int,
        dt_segment_dim: int,
        csc_segment_dim: int,
        gem_segment_dim: int,
        rpc_hit_dim: int | None,
        gem_hit_dim: int | None,
        # output dimensions
        output_dim: int,
        # model hyperparameters
        model_dim: int,
        num_heads: int,
        encoder_num_layers: int,
        decoder_num_layers: int,
        dropout: float = 0.1,
        widening_factor: int = 4,
    ) -> None:
        super().__init__()

        # tracker track: tt
        self.tracker_track_embedder = nn.Linear(
            in_features=tracker_track_dim, out_features=model_dim
        )

        # muon detector measurements: mdm
        self.dt_segment_embedder = nn.Linear(
            in_features=dt_segment_dim, out_features=model_dim
        )
        self.csc_segment_embedder = nn.Linear(
            in_features=csc_segment_dim, out_features=model_dim
        )
        self.gem_segment_embedder = (
            nn.Linear(in_features=gem_segment_dim, out_features=model_dim)
            if gem_segment_dim is not None
            else None
        )
        self.rpc_hit_embedder = (
            nn.Linear(in_features=rpc_hit_dim, out_features=model_dim)
            if rpc_hit_dim is not None
            else None
        )
        self.gem_hit_embedder = (
            nn.Linear(in_features=gem_hit_dim, out_features=model_dim)
            if gem_hit_dim is not None
            else None
        )

        # muon detector measurement encoder: self-attention over ~88 segments
        self.muon_det_encoder = TransformerEncoder(
            model_dim=model_dim,
            num_heads=num_heads,
            num_layers=encoder_num_layers,
            widening_factor=widening_factor,
            dropout=dropout,
        )

        # tracker track decoder: cross-attention to muon memory, no self-attention
        self.decoder = TransformerDecoder(
            model_dim=model_dim,
            num_heads=num_heads,
            num_layers=decoder_num_layers,
            widening_factor=widening_factor,
            dropout=dropout,
            self_attn=False,
        )

        # tracker track-wise classification head
        self.head = nn.Sequential(
            nn.LayerNorm(
                normalized_shape=model_dim,
            ),
            nn.Linear(
                in_features=model_dim,
                out_features=output_dim,
            ),
        )

    def forward(
        self,
        tracker_track: Tensor,
        tracker_track_data_mask: Tensor,
        dt_segment: Tensor,
        dt_segment_data_mask: Tensor,
        csc_segment: Tensor,
        csc_segment_data_mask: Tensor,
        gem_segment: Tensor | None = None,
        gem_segment_data_mask: Tensor | None = None,
        rpc_hit: Tensor | None = None,
        rpc_hit_data_mask: Tensor | None = None,
        gem_hit: Tensor | None = None,
        gem_hit_data_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            tracker_track: (N, L_trk, D_trk)
            tracker_track_data_mask: (N, L_trk)
            dt_segment: (N, L_dt_seg, D_dt_seg)
            dt_segment_data_mask: (N, L_dt_seg)
            csc_segment: (N, L_csc_seg, D_csc_seg)
            csc_segment_data_mask: (N, L_csc_seg)
            gem_segment: (N, L_gem_seg, D_gem_seg)
            gem_segment_data_mask: (N, L_gem_seg)
            rpc_hit: (N, L_rpc_rec, D_rpc_rec)
            rpc_hit_data_mask: (N, L_rpc_rec)
            gem_hit: (N, L_gem_rec, D_gem_rec)
            gem_hit_data_mask: (N, L_gem_rec)

        Returns:
            logits: (N, L_trk)
        """
        # NOTE: projection
        tracker_track_embed = self.tracker_track_embedder(tracker_track)
        dt_segment_embed = self.dt_segment_embedder(dt_segment)
        csc_segment_embed = self.csc_segment_embedder(csc_segment)
        gem_segment_embed = (
            self.gem_segment_embedder(gem_segment) if gem_segment is not None else None
        )
        rpc_hit_embed = self.rpc_hit_embedder(rpc_hit) if rpc_hit is not None else None
        gem_hit_embed = self.gem_hit_embedder(gem_hit) if gem_hit is not None else None

        # NOTE: combine muon detector measurements
        # embed: (N, L_muon_det, D_model)
        # where L_muon_det = L_dt_seg + L_csc_seg + L_gem_seg + L_rpc_hit + L_gem_hit
        muon_det_embed = [dt_segment_embed, csc_segment_embed]
        muon_det_data_mask = [dt_segment_data_mask, csc_segment_data_mask]

        if gem_segment_embed is not None:
            muon_det_embed.append(gem_segment_embed)
            if gem_segment_data_mask is not None:
                muon_det_data_mask.append(gem_segment_data_mask)

        if rpc_hit_embed is not None:
            muon_det_embed.append(rpc_hit_embed)
            if rpc_hit_data_mask is not None:
                muon_det_data_mask.append(rpc_hit_data_mask)

        if gem_hit_embed is not None:
            muon_det_embed.append(gem_hit_embed)
            if gem_hit_data_mask is not None:
                muon_det_data_mask.append(gem_hit_data_mask)

        # concatenate tensors along sequence length dimension
        muon_det_embed = torch.cat(tensors=muon_det_embed, dim=1)
        muon_det_data_mask = torch.cat(tensors=muon_det_data_mask, dim=1)

        # NOTE: muon detector measurement encoding (self-attention)
        muon_det_embed = self.muon_det_encoder(
            input=muon_det_embed,
            data_mask=muon_det_data_mask,
        )

        # NOTE: per-track cross-attention to muon memory (no track self-attention)
        tracker_track_embed = self.decoder(
            target=tracker_track_embed,
            source=muon_det_embed,
            target_data_mask=tracker_track_data_mask,
            source_data_mask=muon_det_data_mask,
        )

        # NOTE: classification head
        #
        # classification head: (N, L_trk, D_model) -> (N, L_trk, 1)
        tracker_track_embed: Tensor = self.head(tracker_track_embed)
        # squeeze: (N, L_trk, 1) -> (N, L_trk)
        tracker_track_embed = tracker_track_embed.squeeze(dim=2)
        # zero out logits at padded track positions
        tracker_track_embed = tracker_track_embed.masked_fill(
            mask=tracker_track_data_mask.logical_not(), value=0
        )
        return tracker_track_embed
