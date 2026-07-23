from collections.abc import Sequence
import math

import torch
from torch import Tensor
from torchmetrics import Metric


VALIDATION_SCORE_THRESHOLDS = 100_001

# pT bin edges shared by the binned operating-point metric, the pT-balanced
# loss weighting, and the differential evaluation plots. Overridden from
# ``config/eval/default.yaml``.
DEFAULT_PT_EDGES: tuple[float, ...] = (
    0.0,
    1.0,
    2.0,
    3.0,
    4.0,
    5.0,
    7.0,
    10.0,
    15.0,
    20.0,
    30.0,
    50.0,
    100.0,
    math.inf,
)


def _format_edge(value: float) -> str:
    if math.isinf(value):
        return "inf"
    if float(value).is_integer():
        return str(int(value))
    return str(value).replace(".", "p")


def format_pt_bin_labels(edges: Sequence[float]) -> list[str]:
    """Aim/JSON-safe names for the pT bins defined by ``edges``."""
    return [
        f"{_format_edge(low)}_{_format_edge(up)}"
        for low, up in zip(edges[:-1], edges[1:])
    ]


class BinnedBinarySpecificityAtSensitivity(Metric):
    """Specificity at sensitivity using an O(samples + thresholds) histogram."""

    full_state_update = False

    def __init__(self, min_sensitivity: float, thresholds: int) -> None:
        super().__init__()
        if not 0.0 <= min_sensitivity <= 1.0:
            raise ValueError("min_sensitivity must be in the [0, 1] range.")
        if thresholds < 2:
            raise ValueError("thresholds must be at least 2.")

        self.min_sensitivity = min_sensitivity
        self.num_thresholds = thresholds
        self.register_buffer(
            "thresholds",
            torch.linspace(0.0, 1.0, thresholds),
            persistent=False,
        )
        self.add_state(
            "histogram",
            default=torch.zeros((2, thresholds), dtype=torch.int64),
            dist_reduce_fx="sum",
        )

    def update(self, preds: Tensor, target: Tensor) -> None:
        validate_binary_metric_inputs(preds=preds, target=target)
        if ((preds < 0.0) | (preds > 1.0)).any():
            raise ValueError("Validation predictions must be in the [0, 1] range.")

        bin_indices = torch.bucketize(preds.contiguous(), self.thresholds, right=True)
        bin_indices = (bin_indices - 1).clamp_(0, self.num_thresholds - 1)
        class_bin_indices = target.long() * self.num_thresholds + bin_indices
        counts = torch.bincount(
            class_bin_indices,
            minlength=2 * self.num_thresholds,
        ).reshape(2, self.num_thresholds)
        self.histogram += counts

    def compute(self) -> tuple[Tensor, Tensor]:
        negative_count = self.histogram[0].sum()
        positive_count = self.histogram[1].sum()
        if positive_count == 0:
            raise ValueError("Specificity at sensitivity requires positive targets.")
        if negative_count == 0:
            raise ValueError("Specificity at sensitivity requires negative targets.")

        true_positives = self.histogram[1].flip(0).cumsum(0).flip(0)
        false_positives = self.histogram[0].flip(0).cumsum(0).flip(0)
        sensitivity = true_positives.float() / positive_count
        specificity = 1.0 - false_positives.float() / negative_count

        eligible = sensitivity >= self.min_sensitivity
        max_specificity = specificity[eligible].max()
        # Match TorchMetrics tie-breaking: choose highest eligible threshold.
        best_index = torch.where(eligible & (specificity == max_specificity))[0][-1]
        return max_specificity, self.thresholds[best_index]


class PtBinnedOperatingPoint(Metric):
    """Per-pT-bin true negative rate at the single global operating point.

    The global threshold is the one ``BinnedBinarySpecificityAtSensitivity``
    would return, computed from the pT-summed histogram. The per-bin rates are
    then read off *at that same threshold*, so they describe the differential
    behaviour of one global cut rather than twelve independently optimized
    ones. ``tnr_macro_pt`` is their unweighted mean and is the secondary
    model-selection metric: the global TNR is dominated by the low-pT bins,
    which hold >99.9% of the negatives, and is therefore blind to the high-pT
    region.

    State is ``(n_pt_bins, 2, thresholds)`` and is fixed by the binning, not by
    the number of evaluated tracks (see ``docs/metric.md``).
    """

    full_state_update = False

    def __init__(
        self,
        min_sensitivity: float,
        thresholds: int,
        pt_edges: Sequence[float] = DEFAULT_PT_EDGES,
    ) -> None:
        super().__init__()
        if not 0.0 <= min_sensitivity <= 1.0:
            raise ValueError("min_sensitivity must be in the [0, 1] range.")
        if thresholds < 2:
            raise ValueError("thresholds must be at least 2.")

        edges = torch.as_tensor(list(pt_edges), dtype=torch.float64)
        if edges.ndim != 1 or edges.numel() < 2:
            raise ValueError("pt_edges must be a 1-D sequence of at least 2 edges.")
        if not bool((edges[1:] > edges[:-1]).all()):
            raise ValueError("pt_edges must be strictly increasing.")

        self.min_sensitivity = min_sensitivity
        self.num_thresholds = thresholds
        self.num_pt_bins = edges.numel() - 1
        self.pt_bin_labels = format_pt_bin_labels([float(each) for each in edges])

        self.register_buffer("pt_edges", edges, persistent=False)
        # inner edges only: ``bucketize`` then maps to [0, num_pt_bins - 1],
        # so tracks below the first edge or above the last fall in the end bins.
        self.register_buffer("pt_boundaries", edges[1:-1].contiguous(), persistent=False)
        self.register_buffer(
            "thresholds",
            torch.linspace(0.0, 1.0, thresholds),
            persistent=False,
        )
        self.add_state(
            "histogram",
            default=torch.zeros(
                (self.num_pt_bins, 2, thresholds),
                dtype=torch.int64,
            ),
            dist_reduce_fx="sum",
        )

    def update(self, preds: Tensor, target: Tensor, pt: Tensor) -> None:
        validate_binary_metric_inputs(preds=preds, target=target)
        if ((preds < 0.0) | (preds > 1.0)).any():
            raise ValueError("Validation predictions must be in the [0, 1] range.")
        if pt.shape != preds.shape:
            raise ValueError(
                f"pt shape {tuple(pt.shape)} does not match "
                f"preds shape {tuple(preds.shape)}."
            )

        score_indices = torch.bucketize(preds.contiguous(), self.thresholds, right=True)
        score_indices = (score_indices - 1).clamp_(0, self.num_thresholds - 1)
        # left-closed bins: pt == edge falls in the bin starting at that edge
        pt_indices = torch.bucketize(
            pt.double().contiguous(), self.pt_boundaries, right=True
        )

        flat_indices = (
            pt_indices * 2 + target.long()
        ) * self.num_thresholds + score_indices
        counts = torch.bincount(
            flat_indices,
            minlength=self.num_pt_bins * 2 * self.num_thresholds,
        ).reshape(self.num_pt_bins, 2, self.num_thresholds)
        self.histogram += counts

    def compute(self) -> dict[str, Tensor]:
        total = self.histogram.sum(dim=0)
        negative_count = total[0].sum()
        positive_count = total[1].sum()
        if positive_count == 0:
            raise ValueError("Specificity at sensitivity requires positive targets.")
        if negative_count == 0:
            raise ValueError("Specificity at sensitivity requires negative targets.")

        true_positives = total[1].flip(0).cumsum(0).flip(0)
        false_positives = total[0].flip(0).cumsum(0).flip(0)
        sensitivity = true_positives.float() / positive_count
        specificity = 1.0 - false_positives.float() / negative_count

        eligible = sensitivity >= self.min_sensitivity
        max_specificity = specificity[eligible].max()
        # Match TorchMetrics tie-breaking: choose highest eligible threshold.
        best_index = torch.where(eligible & (specificity == max_specificity))[0][-1]

        # counts at or above the selected threshold, per pT bin
        selected = self.histogram[..., best_index:].sum(dim=-1)
        per_bin = self.histogram.sum(dim=-1)
        negative_per_bin = per_bin[:, 0]
        positive_per_bin = per_bin[:, 1]

        tnr_pt = torch.where(
            negative_per_bin > 0,
            1.0 - selected[:, 0].float() / negative_per_bin.clamp(min=1),
            torch.full_like(negative_per_bin, float("nan"), dtype=torch.float32),
        )
        tpr_pt = torch.where(
            positive_per_bin > 0,
            selected[:, 1].float() / positive_per_bin.clamp(min=1),
            torch.full_like(positive_per_bin, float("nan"), dtype=torch.float32),
        )

        return {
            "threshold": self.thresholds[best_index],
            "tnr": max_specificity,
            "tpr": sensitivity[best_index],
            "tnr_macro_pt": tnr_pt.nanmean(),
            "tnr_pt": tnr_pt,
            "tpr_pt": tpr_pt,
            "n_negative_pt": negative_per_bin,
            "n_positive_pt": positive_per_bin,
        }


def binary_specificity_at_sensitivity_metric(
    min_sensitivity: float,
) -> BinnedBinarySpecificityAtSensitivity:
    """Build the bounded-memory validation operating-point metric."""
    return BinnedBinarySpecificityAtSensitivity(
        min_sensitivity=min_sensitivity,
        thresholds=VALIDATION_SCORE_THRESHOLDS,
    )


def pt_binned_operating_point_metric(
    min_sensitivity: float,
    pt_edges: Sequence[float] = DEFAULT_PT_EDGES,
) -> PtBinnedOperatingPoint:
    """Build the differential operating-point metric at the same resolution."""
    return PtBinnedOperatingPoint(
        min_sensitivity=min_sensitivity,
        thresholds=VALIDATION_SCORE_THRESHOLDS,
        pt_edges=pt_edges,
    )


def validate_binary_metric_inputs(preds: Tensor, target: Tensor) -> None:
    """Fail close to the source when validation outputs are corrupted."""
    if not preds.is_floating_point():
        raise TypeError(f"Expected floating-point predictions, got {preds.dtype}.")
    if not preds.isfinite().all():
        raise ValueError("Validation predictions contain non-finite values.")

    is_binary = (target == 0) | (target == 1)
    if not is_binary.all():
        invalid_values = target[~is_binary].unique().tolist()
        raise ValueError(
            f"Validation targets contain non-binary values: {invalid_values}."
        )
