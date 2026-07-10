# Ablation Study: Loss Functions for Hard-Positive Emphasis

Status: **planned** (fill in results as runs complete)

This study evaluates the loss functions provided by the config-driven loss
framework (`docs/loss.md`, `config/loss/*.yaml`, `muonly.nn.losses`) against
the BCE baseline. It executes the verification of `docs/plan.md` Priority 1
("Align the loss with the operating point").

## Question

Does concentrating gradient on hard positives — via focal / asymmetric-focal
criteria and/or batch-level tail-targeting auxiliary terms — improve
**TNR at TPR ≥ 99.9%** (`tnr_at_tpr_0p9999`, `docs/metric.md`) over plain
`BCEWithLogitsLoss`?

## Design principles

1. **One knob per phase.** Each phase varies a single factor (criterion,
   focusing strength, aux term, `pos_weight`) with everything else frozen.
2. **Judge only on the metric.** Validation loss values are not comparable
   across criteria (`docs/loss.md`, "Validation semantics"). Model selection
   and all comparisons use validation `tnr_at_tpr_0p9999`.
3. **Pin the seed.** `torch.seed` defaults to `${randbits:}` (random per run).
   All ablation runs pin `torch.seed=20260710` so runs differ only in the loss.
   Seed sensitivity is measured explicitly in the final phase, because a
   99.9%-TPR operating point is statistics-sensitive and single-run
   differences may be noise.
4. **Test set stays untouched** until a winner is selected (metric.md
   evaluation procedure: freeze the validation threshold, then evaluate once
   on test).

## Fixed conditions

| Factor | Value | Source |
| --- | --- | --- |
| Entry point | `uv run python scripts/train.py` (config `no-hit`) | `scripts/train.py` |
| Model | `latent_cross_attention` | `config/default.yaml` |
| Data | `mu2030pu`, no RPC/GEM hits | `config/no-hit.yaml` |
| Optimizer | AdamW-style defaults: lr 3e-4, wd 1e-2, 20 epochs, 5% warmup, cosine annealing, grad clip 1.0 | `config/optim/default.yaml` |
| Batch size | 128 | `config/data_load/default.yaml` |
| Precision | bfloat16 autocast, cuda | `config/torch/default.yaml` |
| Seed | `torch.seed=20260710` (fixed; varied only in Phase 4) | override |
| `pos_weight` | `auto` (= train-set neg/pos ratio) unless the phase varies it | `config/loss/*.yaml` |

Common command prefix (abbreviated below as `TRAIN`):

```bash
TRAIN="uv run python scripts/train.py torch.seed=20260710"
```

Each run writes `result/val.json` (final validation with best checkpoint) and
`sas.json` in its run directory, and tracks `loss`, `loss_base`,
`loss_<aux>`, and validation metrics in Aim. Record the Aim run hash and run
directory in the results tables below.

## Phase 0 — Smoke checks (done once per code change)

`mode=sanity-check` (1024 events, 2 epochs) for `loss=default`, `loss=focal`,
`loss=asymmetric_focal`, `loss=focal_minpos`, `loss=focal_rank`. Pass
criteria: run completes, `tnr_at_tpr_0p9999` computed, score-distribution
plot renders, Aim shows all loss components. Unit-level checks (focal(γ=0) ≡
BCE, ASL(γ,γ,0) ≡ focal(γ), finiteness at logits ±50 in float32/bfloat16,
aux zero-on-empty-side) already pass.

**Status: PASSED (2026-07-10).** All five configs completed end-to-end;
`results/best/{val.json,sas.json,roc.png,eff_pt.png,rej_bkg_pt.png}` written;
Aim tracks `loss`/`loss_base` for all runs plus `loss_minpos` and `loss_rank`
for the aux configs. Run directories (`logs/debug/`): `260710-093545_red-turtle`
(default), `260710-095949_jovial-gaur` (focal), `260710-100004_humorous-gaur`
(asymmetric_focal), `260710-100018_soft-mouse` (focal_minpos),
`260710-100033_diamond-bear` (focal_rank).

## Phase 1 — Criterion ablation

Baseline plus focusing strength, positives-only focusing, and negative
clipping. 8 runs; with 4 GPUs, two batches of 4 (`torch.device=cuda:N`).

| # | Run | Command (append to `TRAIN`) | Rationale |
| --- | --- | --- | --- |
| 1.0 | BCE baseline | `loss=default` | reference |
| 1.1 | focal γ=1 | `loss=focal loss.criterion.gamma=1.0` | mild focusing |
| 1.2 | focal γ=2 | `loss=focal` | standard focal |
| 1.3 | focal γ=3 | `loss=focal loss.criterion.gamma=3.0` | strong focusing |
| 1.4 | ASL γ+=2, γ−=0 | `loss=asymmetric_focal` | focus positives only, negatives plain BCE |
| 1.5 | ASL γ+=3, γ−=0 | `loss=asymmetric_focal loss.criterion.gamma_pos=3.0` | stronger positive focusing |
| 1.6 | ASL γ+=2, γ−=1 | `loss=asymmetric_focal loss.criterion.gamma_neg=1.0` | asymmetric, mild negative focusing |
| 1.7 | ASL γ+=2, γ−=1, clip=0.05 | `loss=asymmetric_focal loss.criterion.gamma_neg=1.0 loss.criterion.clip=0.05` | additionally zero easy negatives |

Results (validation, best checkpoint):

| # | `tnr_at_tpr_0p9999` | threshold | AUROC | run dir / Aim hash |
| --- | --- | --- | --- | --- |
| 1.0 | | | | |
| 1.1 | | | | |
| 1.2 | | | | |
| 1.3 | | | | |
| 1.4 | | | | |
| 1.5 | | | | |
| 1.6 | | | | |
| 1.7 | | | | |

Outcome: pick the best criterion **C\*** (highest `tnr_at_tpr_0p9999`; break
ties toward the simpler criterion). Also note whether the focal family beats
BCE at all — if not, the aux terms in Phase 2 should be tested on top of BCE
instead.

## Phase 2 — Auxiliary batch-level terms

Add tail-targeting aux terms on top of **C\*** (commands below assume
C\* = `focal`; substitute the Phase-1 winner's overrides). Varies term type
and mixing weight; 6 runs.

| # | Run | Command (append to `TRAIN`) |
| --- | --- | --- |
| 2.1 | minpos w=0.1 | `loss=focal_minpos` |
| 2.2 | minpos w=0.3 | `loss=focal_minpos loss.aux.0.weight=0.3` |
| 2.3 | minpos w=0.1, k=64 | `loss=focal_minpos loss.aux.0.k=64` |
| 2.4 | rank w=0.1 | `loss=focal_rank` |
| 2.5 | rank w=0.3 | `loss=focal_rank loss.aux.0.weight=0.3` |
| 2.6 | rank w=0.1, k=128/128 | `loss=focal_rank loss.aux.0.k_pos=128 loss.aux.0.k_neg=128` |

While judging on the metric, also inspect in Aim that `loss_minpos` /
`loss_rank` decrease over training and that their weighted share of `loss`
is neither negligible (<1%, term inert — raise `weight`) nor dominant
(>50%, term hijacks training — lower `weight`).

Results:

| # | `tnr_at_tpr_0p9999` | threshold | aux share of total loss (end of training) | run dir / Aim hash |
| --- | --- | --- | --- | --- |
| 2.1 | | | | |
| 2.2 | | | | |
| 2.3 | | | | |
| 2.4 | | | | |
| 2.5 | | | | |
| 2.6 | | | | |

Outcome: best combination **C\* + A\*** (or no aux term if none beats C\*
alone).

## Phase 3 — `pos_weight` interaction

Focal focusing and class reweighting overlap: the `(1-p_t)^γ` factor already
suppresses the easy-negative flood, so the large `auto` ratio (neg/pos of the
train set) may over-boost positives and cost background rejection. Varies
`pos_weight` for the Phase-2 winner; 3 runs plus the already-available `auto`
result.

| # | Run | Command (append to `TRAIN` + winner overrides) |
| --- | --- | --- |
| 3.0 | auto | (= Phase-2 winner, no new run) |
| 3.1 | pos_weight=1 | `loss.pos_weight=1` |
| 3.2 | pos_weight=10 | `loss.pos_weight=10` |
| 3.3 | pos_weight=50 | `loss.pos_weight=50` |

Results:

| # | `tnr_at_tpr_0p9999` | threshold | run dir / Aim hash |
| --- | --- | --- | --- |
| 3.0 | | | |
| 3.1 | | | |
| 3.2 | | | |
| 3.3 | | | |

## Phase 4 — Seed robustness and final comparison

The 99.9%-TPR threshold is set by a handful of validation tracks, so
single-run margins can be statistical. Rerun the overall winner and the BCE
baseline with 3 seeds each (the pinned `20260710` run already exists for
both):

```bash
$TRAIN <winner overrides> torch.seed=1
$TRAIN <winner overrides> torch.seed=2
$TRAIN loss=default torch.seed=1
$TRAIN loss=default torch.seed=2
```

| Config | seed 20260710 | seed 1 | seed 2 | mean | spread |
| --- | --- | --- | --- | --- | --- |
| BCE baseline | | | | | |
| winner | | | | | |

Declare improvement only if the winner's worst seed beats the baseline's best
seed, or at minimum if the mean gap clearly exceeds the seed spread.

## Final test-set evaluation

For the declared winner only (metric.md procedure):

1. Freeze the score threshold returned by `tnr_at_tpr_0p9999` on validation.
2. Evaluate once on the test set at that fixed threshold; report achieved
   TPR, TNR, FPR, and fraction of tracks sent to extrapolation.
3. Report fixed-threshold performance vs pT, η, pileup; check the
   `loss_pt_0p5_3` / `loss_pt_3_inf` diagnostics for regressions in the
   low-pT bin.

## Bookkeeping

- Total budget: 8 (P1) + 6 (P2) + 3 (P3) + 4 (P4) = 21 full runs at 20
  epochs each; parallelize 4-wide across GPUs via `torch.device=cuda:{0..3}`.
- Every run's exact config is stored by Hydra in its run directory; the
  tables above only need the run directory / Aim hash to be reproducible.
- If two configs tie within seed spread, prefer (in order): fewer
  hyperparameters, no aux term, lower γ.

## Conclusions

_To be filled in after Phase 4 and the test-set evaluation._
