# Loss Functions

This document describes the config-driven loss framework used by
`scripts/train.py` and the motivation behind the loss functions it provides.
It implements Priority 1 of `docs/plan.md` ("Align the loss with the operating
point").

## Motivation

The primary evaluation metric is **TNR at TPR ≥ 99.9%** (`docs/metric.md`).
This operating point is decided entirely by the ~0.1% *hardest positives* —
the signal tracker tracks the model scores lowest set the decision threshold,
and the threshold in turn determines how much background can be rejected.

Plain mean `BCEWithLogitsLoss` is a poor surrogate for this metric: once most
examples are classified confidently, the loss (and its gradient) is dominated
by the bulk of easy examples, and the tail of hard positives that actually
determines the metric receives a vanishing share of the gradient. A model can
improve its average loss or AUROC while regressing at the 99.9%-TPR tail.

The framework therefore provides losses that concentrate gradient on hard
positives, plus auxiliary terms that attack the tail directly, so that
alternatives can be ablated against the BCE baseline by switching a single
Hydra config group.

## Configuration

The loss is selected with the `loss` config group (`config/loss/*.yaml`).
Every loss config has the structure:

```yaml
pos_weight: auto        # "auto" (= neg/pos ratio of the train set) or a number

criterion:              # per-track criterion, hydra-instantiated
  _target_: muonly.nn.BinaryFocalLoss
  gamma: 2.0
  reduction: none       # required: training/validation average it themselves

aux:                    # optional list of batch-level auxiliary terms
  - _target_: muonly.nn.SoftMinPositiveMarginLoss
    margin: 2.0
    k: 16
    weight: 0.1
```

- `pos_weight` is resolved first (dataset scan when `auto`) and injected into
  the criterion at instantiation. All criteria follow the `BCEWithLogitsLoss`
  convention: the positive-class term is multiplied by `pos_weight`, the
  negative-class term by 1.
- The criterion must return a per-element loss (`reduction: none`);
  `compute_loss` masks padded tracks and averages, and `validate` reuses the
  same per-element values for the pT-binned loss metrics.
- Each `aux` entry is a module returning a scalar on the masked in-batch
  logits/targets; it is added to the total as `total += weight * value`.
  Every component is tracked in Aim separately (`loss`, `loss_base`,
  `loss_<name>`).

Available configs:

| Config                 | Criterion                        | Aux term                    |
| ---------------------- | -------------------------------- | --------------------------- |
| `loss=default`         | `torch.nn.BCEWithLogitsLoss`     | —                           |
| `loss=focal`           | `BinaryFocalLoss(gamma=2)`       | —                           |
| `loss=asymmetric_focal`| `AsymmetricFocalLoss(2, 0, 0)`   | —                           |
| `loss=focal_minpos`    | `BinaryFocalLoss(gamma=2)`       | `SoftMinPositiveMarginLoss` |
| `loss=focal_rank`      | `BinaryFocalLoss(gamma=2)`       | `TopKPairwiseRankingLoss`   |

Hyperparameter sweeps need no new files:

```bash
python scripts/train.py loss=focal loss.criterion.gamma=1.0 loss.pos_weight=50
python scripts/train.py loss=asymmetric_focal loss.criterion.gamma_pos=3.0
python scripts/train.py loss=focal_minpos loss.aux.0.weight=0.3
```

All losses live in `src/muonly/nn/losses.py` and are computed in logit space
via `logsigmoid` (no naked `sigmoid().log()`), so they are numerically stable
under the bfloat16 autocast used for training.

## Per-track criteria

### `BinaryFocalLoss`

Focal loss (Lin et al. 2017) on logits:

$$\mathcal{L} = -w_t\,(1 - p_t)^{\gamma}\,\log p_t$$

where $p_t$ is the predicted probability of the true class and
$w_t = \texttt{pos\_weight}$ for positives, 1 for negatives. The
$(1-p_t)^\gamma$ factor suppresses the loss of well-classified examples, so
the gradient concentrates on the hard examples — in particular the hard
positives that set the 99.9%-TPR threshold. `gamma=0` reduces exactly to
`BCEWithLogitsLoss`.

### `AsymmetricFocalLoss`

Asymmetric focal loss (ASL, Ben-Baruch et al. 2020) with independent focusing
exponents for the two classes:

- positives: $-w \,(1-p)^{\gamma_+}\,\log p$
- negatives: $-p^{\gamma_-}\,\log(1-p)$

Note the direction is **opposite to the multilabel ASL paper**: there,
$\gamma_- > \gamma_+$ downweights the flood of easy negatives; here the metric
is set by hard positives, so hard-positive emphasis means
$\gamma_+ > \gamma_-$ — keep full focusing pressure on positives while leaving
the (already `pos_weight`-downweighted) negatives closer to plain BCE.

The optional `clip` parameter $m$ applies ASL's probability shift to the
negative term, $p \to \max(p - m, 0)$, fully zeroing the loss of easy
negatives with $p < m$. Use $\gamma_- \geq 1$ together with $m > 0$ to keep
gradients bounded at $p = m$.

With $\gamma_+ = \gamma_-$ and $m = 0$ this equals `BinaryFocalLoss`.

## Auxiliary batch-level terms

Per-element focal losses still act on each track independently. The two
auxiliary terms act on the *batch score distribution*, targeting the tail
that defines the metric (plan.md P1 items 2 and 4). Both return a scalar and
carry a `weight` (mixing coefficient) and a `name` (Aim logging key).

### `SoftMinPositiveMarginLoss` (`minpos`)

Takes the $\min(k, n_+)$ lowest positive logits in the batch and returns
$\operatorname{softplus}(\text{margin} - x)$ averaged over them — a smooth
hinge that pushes the *worst* positives above `margin` in logit space. This is
the most direct attack on the quantity the metric measures: the location of
the lowest positive scores. Zero when the batch has no positives.

### `TopKPairwiseRankingLoss` (`rank`)

A partial-AUC surrogate. Takes the `k_pos` lowest positive logits and the
`k_neg` highest negative logits and averages the pairwise logistic loss
$\operatorname{softplus}(\text{margin} - (x_+ - x_-))$ over the
$k_+ \times k_-$ grid. This optimizes the ranking exactly in the region where
TNR@TPR≥99.9% is decided — hardest positives vs. hardest negatives — instead
of the global ranking that AUROC-style losses optimize. Zero when either side
is empty in the batch.

## Validation semantics

`validate()` applies only the per-element base criterion (no aux terms), so
the `loss`, `loss_pt_0p5_3`, `loss_pt_3_inf` validation metrics remain
per-track base-criterion means. **Validation loss values are not comparable
across different criteria** — compare runs on `tnr_at_tpr_0p9999`
(see `docs/metric.md`), which is also what the ablation should be judged on.

## Deprecated

The former pT-binned loss balancing (`loss.balancing`, configs
`loss=pt_0_3` / `loss=pt_bin`) was removed together with the hard-coded BCE
criterion when this framework was introduced. Per-pT-bin *diagnostics* remain
available through the validation loss metrics.
