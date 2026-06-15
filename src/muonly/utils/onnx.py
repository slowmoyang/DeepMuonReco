from torch import Tensor
import torch.nn as nn


class Phase2NoHitModelWrapper(nn.Module):

    def __init__(
        self,
        model: nn.Module,
        tracker_track_preprocessing: nn.Module,
        dt_segment_preprocessing: nn.Module,
        csc_segment_preprocessing: nn.Module,
        gem_segment_preprocessing: nn.Module,
    ) -> None:
        super().__init__()
        self.model = model
        self.tracker_track_preprocessing = tracker_track_preprocessing
        self.dt_segment_preprocessing = dt_segment_preprocessing
        self.csc_segment_preprocessing = csc_segment_preprocessing
        self.gem_segment_preprocessing = gem_segment_preprocessing

    def forward(
        self,
        tracker_track: Tensor,
        tracker_track_data_mask: Tensor,
        dt_segment: Tensor,
        dt_segment_data_mask: Tensor,
        csc_segment: Tensor,
        csc_segment_data_mask: Tensor,
        gem_segment: Tensor,
        gem_segment_data_mask: Tensor,
    ) -> Tensor:
        tracker_track = self.tracker_track_preprocessing(tracker_track)
        dt_segment = self.dt_segment_preprocessing(dt_segment)
        csc_segment = self.csc_segment_preprocessing(csc_segment)
        gem_segment = self.gem_segment_preprocessing(gem_segment)

        logits = self.model(
            tracker_track=tracker_track,
            tracker_track_data_mask=tracker_track_data_mask,
            dt_segment=dt_segment,
            dt_segment_data_mask=dt_segment_data_mask,
            csc_segment=csc_segment,
            csc_segment_data_mask=csc_segment_data_mask,
            gem_segment=gem_segment,
            gem_segment_data_mask=gem_segment_data_mask,
            rpc_hit=None,
            rpc_hit_data_mask=None,
            gem_hit=None,
            gem_hit_data_mask=None,
        )
        return logits
