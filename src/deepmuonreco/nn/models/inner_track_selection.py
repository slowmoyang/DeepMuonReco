from torch import Tensor
import torch.nn as nn
from ..utils import make_cross_attn_mask, make_self_attn_mask
from deepmuonreco.data import InnerTrackSelectionDataset


class InnerTrackSelectionTransformer(nn.Module):

    def __init__(
        self,
        dim_track: int = InnerTrackSelectionDataset.DIM_TRACK,
        dim_segment: int = InnerTrackSelectionDataset.DIM_SEGMENT,
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

        self.track_embedder = nn.Linear(
            in_features=dim_track,
            out_features=dim_model,
        )

        self.segment_embedder = nn.Linear(
            in_features=dim_segment,
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
    ) -> Tensor:
        """
        Args:
            track: (N, L_trk, D_trk)
            track_pad_mask: (N, L_trk)
            segment: (N, L_seg, D_seg)
            segment_pad_mask: (N, L_seg)
        Shape:
            N:batch size
            L_trk: the max track length in a batch
            D_trk: track dimension (==len([pt, eta, phi]))
            L_seg: the max segment length in a batch,
            D_seg: segment dimension (==6)
        """
        # track: (N, L_trk, D_trk) -> (N, L_trk, D_model)
        track_embed = self.track_embedder(track)
        # segment: (N, L_seg, D_seg) -> (N, L_seg, D_model)
        segment_embed = self.segment_embedder(segment)

        # compute tgt_mask using track_pad_mask
        tgt_mask = make_self_attn_mask(
            pad_mask=track_pad_mask,
            num_heads=self.num_heads,
        )

        memory_mask = make_cross_attn_mask(
            source_pad_mask=segment_pad_mask,
            target_pad_mask=track_pad_mask,
            num_heads=self.num_heads,
        )

        # latent: tracks in the latent space
        # latent: (N, L_trk, D_model)
        latent = self.backbone(
            tgt=track_embed, # target
            memory=segment_embed,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            memory_key_padding_mask=segment_pad_mask,
            tgt_key_padding_mask=track_pad_mask,
            tgt_is_causal=False,
            memory_is_causal=False,
        )

        # latent: (N, L_trk, D_model) -> (N, L_trk, D_output), where D_output==1
        logits: Tensor = self.classification_head(latent)
        # (N, L_trk, 1) -> (N, L_trk)
        logits = logits.squeeze(dim=2)

        logits = logits.masked_fill(mask=track_pad_mask, value=0)
        return logits
