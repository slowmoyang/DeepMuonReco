import torch
from torch import Tensor
from torchmetrics import Metric


VALIDATION_SCORE_THRESHOLDS = 100_001


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


def binary_specificity_at_sensitivity_metric(
    min_sensitivity: float,
) -> BinnedBinarySpecificityAtSensitivity:
    """Build the bounded-memory validation operating-point metric."""
    return BinnedBinarySpecificityAtSensitivity(
        min_sensitivity=min_sensitivity,
        thresholds=VALIDATION_SCORE_THRESHOLDS,
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
