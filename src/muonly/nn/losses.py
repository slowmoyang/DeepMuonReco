import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


__all__ = [
    "BinaryFocalLoss",
    "AsymmetricFocalLoss",
    "SoftMinPositiveMarginLoss",
    "TopKPairwiseRankingLoss",
]


def _reduce(loss: Tensor, reduction: str) -> Tensor:
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")


class BinaryFocalLoss(nn.Module):
    """Focal loss on logits for binary classification.

    ``loss = -w_t * (1 - p_t)^gamma * log(p_t)`` where ``p_t`` is the predicted
    probability of the true class. ``gamma=0`` reduces exactly to
    ``BCEWithLogitsLoss``. Computed in logit space via ``logsigmoid`` for
    numerical stability under autocast.

    ``pos_weight`` follows the ``BCEWithLogitsLoss`` convention: the positive
    term is multiplied by ``pos_weight``, the negative term by 1 (no
    alpha/(1-alpha) parameterization).
    """

    def __init__(
        self,
        gamma: float = 2.0,
        pos_weight: Tensor | None = None,
        reduction: str = "none",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        input = input.float()
        target = target.float()

        z = input * (2 * target - 1)
        log_pt = F.logsigmoid(z)
        log_one_minus_pt = F.logsigmoid(-z)

        loss = -torch.exp(self.gamma * log_one_minus_pt) * log_pt
        if self.pos_weight is not None:
            loss = loss * (1 + (self.pos_weight - 1) * target)
        return _reduce(loss, self.reduction)

    def extra_repr(self) -> str:
        return f"gamma={self.gamma}, reduction={self.reduction}"


class AsymmetricFocalLoss(nn.Module):
    """Asymmetric focal loss (ASL, Ben-Baruch et al. 2020) on logits.

    Positives and negatives get independent focusing exponents. Unlike the
    multilabel ASL setting (``gamma_neg > gamma_pos``), hard-positive emphasis
    for the TNR@TPR>=99.9% operating point means ``gamma_pos > gamma_neg``.

    ``clip`` (probability shift ``m``) softens the negative term: the negative
    probability is shifted down by ``m`` before focusing, fully zeroing the
    loss of easy negatives with ``p < m``. Use ``gamma_neg >= 1`` with
    ``clip > 0`` to keep gradients bounded at ``p == m``.

    ``pos_weight`` follows the ``BCEWithLogitsLoss`` convention. With
    ``gamma_pos == gamma_neg`` and ``clip == 0`` this equals
    ``BinaryFocalLoss``.
    """

    def __init__(
        self,
        gamma_pos: float = 2.0,
        gamma_neg: float = 0.0,
        clip: float = 0.0,
        pos_weight: Tensor | None = None,
        reduction: str = "none",
    ) -> None:
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.reduction = reduction
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        input = input.float()
        target = target.float()

        log_p = F.logsigmoid(input)
        log_one_minus_p = F.logsigmoid(-input)

        pos_loss = -torch.exp(self.gamma_pos * log_one_minus_p) * log_p
        if self.pos_weight is not None:
            pos_loss = self.pos_weight * pos_loss

        if self.clip > 0:
            # probability shift: log argument is bounded below by ``clip``
            p_shifted = (input.sigmoid() - self.clip).clamp(min=0)
            one_minus_p_shifted = ((-input).sigmoid() + self.clip).clamp(max=1.0)
            neg_loss = -(p_shifted**self.gamma_neg) * one_minus_p_shifted.log()
        else:
            neg_loss = -torch.exp(self.gamma_neg * log_p) * log_one_minus_p

        loss = torch.where(target > 0.5, pos_loss, neg_loss)
        return _reduce(loss, self.reduction)

    def extra_repr(self) -> str:
        return (
            f"gamma_pos={self.gamma_pos}, gamma_neg={self.gamma_neg}, "
            f"clip={self.clip}, reduction={self.reduction}"
        )


class SoftMinPositiveMarginLoss(nn.Module):
    """Auxiliary term pushing the worst positive logits above a margin.

    Selects the ``min(k, n_pos)`` lowest positive logits per batch and returns
    ``softplus(margin - logit).mean()`` — a smooth hinge that directly attacks
    the hardest positives which set the TPR>=99.9% threshold. Returns a scalar;
    zero if the batch has no positives.

    ``weight`` and ``name`` are read by the training loop for weighting the
    total loss and for logging.
    """

    def __init__(
        self,
        margin: float = 2.0,
        k: int = 16,
        weight: float = 0.1,
        name: str = "minpos",
    ) -> None:
        super().__init__()
        self.margin = margin
        self.k = k
        self.weight = weight
        self.name = name

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        pos_logits = input[target > 0.5].float()
        if pos_logits.numel() == 0:
            return input.new_zeros((), dtype=torch.float32)

        k = min(self.k, pos_logits.numel())
        lowest = pos_logits.topk(k, largest=False).values
        return F.softplus(self.margin - lowest).mean()

    def extra_repr(self) -> str:
        return f"margin={self.margin}, k={self.k}, weight={self.weight}"


class TopKPairwiseRankingLoss(nn.Module):
    """Auxiliary partial-AUC surrogate on the hardest in-batch pairs.

    Pairwise logistic loss between the ``k_pos`` lowest positive logits and the
    ``k_neg`` highest negative logits:
    ``softplus(margin - (x_pos - x_neg)).mean()`` over the pair grid. Optimizes
    the ranking exactly where the TNR@TPR>=99.9% metric is decided. Returns a
    scalar; zero if the batch lacks positives or negatives.

    ``weight`` and ``name`` are read by the training loop for weighting the
    total loss and for logging.
    """

    def __init__(
        self,
        k_pos: int = 32,
        k_neg: int = 32,
        margin: float = 1.0,
        weight: float = 0.1,
        name: str = "rank",
    ) -> None:
        super().__init__()
        self.k_pos = k_pos
        self.k_neg = k_neg
        self.margin = margin
        self.weight = weight
        self.name = name

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        input = input.float()
        pos_logits = input[target > 0.5]
        neg_logits = input[target <= 0.5]
        if pos_logits.numel() == 0 or neg_logits.numel() == 0:
            return input.new_zeros((), dtype=torch.float32)

        hardest_pos = pos_logits.topk(
            min(self.k_pos, pos_logits.numel()), largest=False
        ).values
        hardest_neg = neg_logits.topk(
            min(self.k_neg, neg_logits.numel()), largest=True
        ).values

        pair_margin = hardest_pos[:, None] - hardest_neg[None, :]
        return F.softplus(self.margin - pair_margin).mean()

    def extra_repr(self) -> str:
        return (
            f"k_pos={self.k_pos}, k_neg={self.k_neg}, "
            f"margin={self.margin}, weight={self.weight}"
        )
