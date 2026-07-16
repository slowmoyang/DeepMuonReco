# Ablation Study: Loss Functions for Hard-Positive Emphasis

Status: **in progress** (Phase 1: rerun at 100 epochs, 8/8 runs evaluated,
winner selected)

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
| Optimizer | AdamW-style defaults: lr 3e-4, wd 1e-2, 5% warmup, cosine annealing, grad clip 1.0 | `config/optim/default.yaml` |
| Epochs | 100 (`optim.max_epochs=100`; the default 20-epoch schedule was found to change the criterion ranking, see Phase 1) | override |
| Batch size | 128 | `config/data_load/default.yaml` |
| Precision | bfloat16 autocast, cuda | `config/torch/default.yaml` |
| Seed | `torch.seed=20260710` (fixed; varied only in Phase 4) | override |
| `pos_weight` | `auto` (= train-set neg/pos ratio) unless the phase varies it | `config/loss/*.yaml` |

Common command prefix (abbreviated below as `TRAIN`):

```bash
TRAIN="uv run python scripts/train.py torch.seed=20260710 optim.max_epochs=100"
```

The BCE baseline config is `loss=bce` (`config/loss/bce.yaml`); the former
`loss=default` name no longer exists. Phase-1 runs were launched via
`submit/phase1.fish` with `exp=phase1` and explicit `run=` directory names.

Each run writes `result/val.json` (final validation with best checkpoint) and
`sas.json` in its run directory, and tracks `loss`, `loss_base`,
`loss_<aux>`, and validation metrics in Aim. Record the Aim run hash and run
directory in the results tables below.

## Phase 0 — Smoke checks (done once per code change)

`mode=sanity-check` (1024 events, 2 epochs) for `loss=bce` (formerly
`loss=default`), `loss=focal`,
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
| 1.0 | BCE baseline | `loss=bce` | reference |
| 1.1 | focal γ=1 | `loss=focal loss.criterion.gamma=1.0` | mild focusing |
| 1.2 | focal γ=2 | `loss=focal` | standard focal |
| 1.3 | focal γ=3 | `loss=focal loss.criterion.gamma=3.0` | strong focusing |
| 1.4 | ASL γ+=2, γ−=0 | `loss=asymmetric_focal` | focus positives only, negatives plain BCE |
| 1.5 | ASL γ+=3, γ−=0 | `loss=asymmetric_focal loss.criterion.gamma_pos=3.0` | stronger positive focusing |
| 1.6 | ASL γ+=2, γ−=1 | `loss=asymmetric_focal loss.criterion.gamma_neg=1.0` | asymmetric, mild negative focusing |
| 1.7 | ASL γ+=2, γ−=1, clip=0.05 | `loss=asymmetric_focal loss.criterion.gamma_neg=1.0 loss.criterion.clip=0.05` | additionally zero easy negatives |

Results at 100 epochs (validation, best checkpoint; `sas.json` entry
`tpr_requested=0.9999`, AUROC from `val.json`):

| # | `tnr_at_tpr_0p9999` | threshold | AUROC | run dir / Aim hash |
| --- | --- | --- | --- | --- |
| 1.0 | 0.332205 | 0.001648 | 0.918929 | `logs/phase1/run-00_bce_00` |
| 1.1 | 0.334011 | 0.030640 | 0.913302 | `logs/phase1/run-01_focal_gamma-1p0_00` |
| 1.2 | 0.353297 | 0.088867 | 0.917017 | `logs/phase1/run-02_focal_gamma-2p0_00` |
| 1.3 | **0.370795** | 0.142578 | 0.920179 | `logs/phase1/run-03_focal_gamma-3p0_00` |
| 1.4 | 0.325054 | 0.001457 | 0.910540 | `logs/phase1/run-04_asymmetric-focal_gamma-pos-2p0_gamma-neg-0p0_00` |
| 1.5 | 0.217603 | 0.001503 | 0.894176 | `logs/phase1/run-05_asymmetric-focal_gamma-pos-3p0_gamma-neg-0p0_00` |
| 1.6 | 0.342389 | 0.041992 | 0.909713 | `logs/phase1/run-06_asymmetric-focal_gamma-pos-2p0_gamma-neg-1p0_00` |
| 1.7 | 0.367474 | 0.048828 | 0.919480 | `logs/phase1/run-07_asymmetric-focal_gamma-pos-2p0_gamma-neg-1p0_clip-0p05_00` |

**Status (2026-07-16):** All eight runs completed at 100 epochs. The
Phase-1 winner **C\*** is focal γ=3 (run 1.3), with TNR 0.370795 at the
target operating point. Clipped ASL γ+=2, γ−=1 (run 1.7) is a close second
at 0.367474 (Δ = 0.003321); the simplicity tie-break also favors focal.

Outcome: pick the best criterion **C\*** (highest `tnr_at_tpr_0p9999`; break
ties toward the simpler criterion). Also note whether the focal family beats
BCE at all — if not, the aux terms in Phase 2 should be tested on top of BCE
instead. Both conditions resolve in focal's favor at 100 epochs.

### Superseded 20-epoch results

The matrix was first run at the default 20-epoch schedule. Because these
results and the uncontrolled 200-epoch `loss-ablation` runs disagreed on the
leader, the matrix was rerun at 100 epochs (tables above); the rerun
overwrote the original `logs/phase1/run-*` directories, so the 20-epoch
numbers below are preserved here only for the record.

| # | `tnr_at_tpr_0p9999` | threshold | AUROC |
| --- | --- | --- | --- |
| 1.0 | 0.108602 | 0.018311 | 0.812964 |
| 1.1 | 0.099902 | 0.096680 | 0.802574 |
| 1.2 | 0.137747 | 0.157227 | 0.814449 |
| 1.3 | 0.135111 | 0.221680 | 0.807183 |
| 1.4 | 0.128174 | 0.015442 | 0.786776 |
| 1.5 | **0.143639** | 0.017944 | 0.782773 |
| 1.6 | 0.135073 | 0.080566 | 0.801179 |
| 1.7 | 0.142550 | 0.104980 | 0.813670 |

At 20 epochs the winner was ASL γ+=3, γ−=0 (run 1.5, TNR 0.143639) — the
opposite end of the 100-epoch ranking, where the same config is worst. See
the Phase 1 conclusion.

### Additional 200-epoch loss-ablation runs

The separate `logs/loss-ablation/` runs used `optim.max_epochs=200` and
selected the best validation checkpoint over that longer schedule. They are
contextual results rather than a controlled extension of Phase 1: seeds were
left at `torch.seed=${randbits:}`, and the loss configurations differ. The
matrix covers BCE; ASL γ+=3, γ−=0; focal γ=1; focal γ=1 with
`pos_weight=50`; and focal γ=2 plus minpos with auxiliary weight 0.3.

Results (validation, best checkpoint):

| Run | `tnr_at_tpr_0p9999` | threshold | AUROC |
| --- | --- | --- | --- |
| `asymmetric-focal_gamma-pos-3p0_00` | 0.365657 | 0.001167 | 0.924513 |
| `asymmetric-focal_gamma-pos-3p0_01` | 0.369624 | 0.000710 | 0.927484 |
| `asymmetric-focal_gamma-pos-3p0_02` | 0.399277 | 0.002258 | 0.926901 |
| `bce_00` | 0.399300 | 0.000938 | 0.933311 |
| `bce_01` | 0.383804 | 0.000805 | 0.930611 |
| `bce_02` | 0.424703 | 0.000969 | 0.937177 |
| `focal-minpos_aux-0-weight-0p3_00` | 0.000007 | 0.558594 | 0.750243 |
| `focal-minpos_aux-0-weight-0p3_02` | 0.000001 | 0.554688 | 0.767235 |
| `focal_gamma-1p0_00` | 0.409049 | 0.014038 | 0.930346 |
| `focal_gamma-1p0_01` | 0.411936 | 0.016357 | 0.933171 |
| `focal_gamma-1p0_02` | 0.411412 | 0.018311 | 0.932918 |
| `focal_gamma-1p0_pos-weight-50_00` | **0.455428** | 0.021973 | **0.941163** |
| `focal_gamma-1p0_pos-weight-50_01` | 0.430989 | 0.024048 | 0.936734 |

Thirteen of the fifteen run directories contain a final `sas.json`.
`focal-minpos_aux-0-weight-0p3_01` and
`focal_gamma-1p0_pos-weight-50_02` have no final result. Among available
runs, focal γ=1 with `pos_weight=50` is strongest; its two completed
replicas reach TNR 0.455428 and 0.430989. The three focal γ=1 replicas are
stable at 0.409049–0.411936, while the minpos runs have near-zero TNR. Because
training length, seed, and loss hyperparameters differ, these values should
not be used to select **C\*** from the controlled Phase-1 table.

### Phase 1 conclusion

At seed `20260710` and 100 epochs, symmetric focal focusing improves the
target operating-point metric over BCE, and focusing strength helps
monotonically: γ=1 → γ=2 → γ=3 gives 0.334011 → 0.353297 → 0.370795. The
winner **C\*** is focal γ=3, with an absolute gain of 0.038590 over BCE
(11.6% relative). Clipped ASL γ+=2, γ−=1 (run 1.7) trails by only
0.003321, but the tie-break rule (fewer hyperparameters, lower complexity)
selects focal.

The criterion ranking is **not stable in training length**. The 20-epoch
winner, ASL γ+=3, γ−=0, collapses to the worst result at 100 epochs
(0.217603, well below BCE) — consistent with the uncontrolled 200-epoch
runs, where ASL γ+=3 also trailed BCE. Positives-only focusing appears to
help early in training but hurt once the model converges further, so
short-schedule ablations of this loss family are unreliable; all remaining
phases run at 100 epochs.

The AUROC ordering again does not match the target-metric ordering
(e.g. run 1.6 beats run 1.1 on TNR but not on AUROC), reinforcing that
**C\*** must be selected using `tnr_at_tpr_0p9999` rather than AUROC.

## Phase 2 — Auxiliary batch-level terms

Add tail-targeting aux terms on top of **C\***. The `focal_minpos` /
`focal_rank` configs already use `BinaryFocalLoss` (γ=2 default), so the
selected focal γ=3 criterion needs a single override:

```bash
CSTAR=(loss.criterion.gamma=3.0)
```

The phase varies term type and mixing weight; 6 runs.

| # | Run | Command (append to `TRAIN`) |
| --- | --- | --- |
| 2.1 | minpos w=0.1 | `loss=focal_minpos "${CSTAR[@]}"` |
| 2.2 | minpos w=0.3 | `loss=focal_minpos "${CSTAR[@]}" loss.aux.0.weight=0.3` |
| 2.3 | minpos w=0.1, k=64 | `loss=focal_minpos "${CSTAR[@]}" loss.aux.0.k=64` |
| 2.4 | rank w=0.1 | `loss=focal_rank "${CSTAR[@]}"` |
| 2.5 | rank w=0.3 | `loss=focal_rank "${CSTAR[@]}" loss.aux.0.weight=0.3` |
| 2.6 | rank w=0.1, k=128/128 | `loss=focal_rank "${CSTAR[@]}" loss.aux.0.k_pos=128 loss.aux.0.k_neg=128` |

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

Focal/asymmetric focusing and class reweighting overlap: the focusing factors
already suppress easy examples, so the large `auto` ratio (neg/pos of the
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
$TRAIN loss=bce torch.seed=1
$TRAIN loss=bce torch.seed=2
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

- Total budget: 8 (P1) + 6 (P2) + 3 (P3) + 4 (P4) = 21 full runs at 100
  epochs each; parallelize 4-wide across GPUs via `torch.device=cuda:{0..3}`.
- Every run's exact config is stored by Hydra in its run directory; the
  tables above only need the run directory / Aim hash to be reproducible.
- If two configs tie within seed spread, prefer (in order): fewer
  hyperparameters, no aux term, lower γ.

## Conclusions

_To be filled in after Phase 4 and the test-set evaluation._
