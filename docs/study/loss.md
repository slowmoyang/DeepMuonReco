# Ablation Study: Loss Functions for Hard-Positive Emphasis

Status: **model selection complete** (focal γ=3 with `pos_weight=100`
beats BCE at all six matched seeds and is selected for the final model;
test-set performance evaluation pending)

This study evaluates the loss functions provided by the config-driven loss
framework (`docs/loss.md`, `config/loss/*.yaml`, `muonly.nn.losses`) against
the BCE baseline. It executes the verification of `docs/plan.md` Priority 1
("Align the loss with the operating point").

## Phase goals

| Phase | Goal |
| --- | --- |
| 0 — Smoke checks | Verify that every loss configuration runs end to end and that its metrics, plots, and loss components are recorded correctly. |
| 1 — Criterion ablation | Isolate the effect of the base criterion and focusing strength, then select the best criterion **C\*** at the target operating point. |
| 2 — Auxiliary terms | Test whether batch-level hard-positive terms improve **C\***, then select the best auxiliary setup **A\*** (including no auxiliary term). |
| 3 — `pos_weight` interaction | Tune class reweighting on top of **C\* + A\*** and select the overall candidate configuration. |
| 4 — Seed robustness | Measure seed sensitivity, compare the candidate with BCE across matched seeds, and finalize model selection before test-set evaluation. |

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

Results (validation, best checkpoint; aux share = `weight` ×
`loss_<aux>` / `loss` at the final training epoch, from Aim):

| # | `tnr_at_tpr_0p9999` | threshold | AUROC | aux share (end) | run dir / Aim hash |
| --- | --- | --- | --- | --- | --- |
| 2.1 | 0.087907 | 0.285156 | 0.813709 | 61.6% | `logs/phase2/run-01_focal-minpos_weight-0p1_00` / `b72b28726d1d4b4fa4037c12` |
| 2.2 | 0.000000 | 0.589844 | 0.500000 | 71.9% | `logs/phase2/run-02_focal-minpos_weight-0p3_00` / `d5ec9b3287334914965e6bb8` |
| 2.3 | **0.124055** | 0.333984 | 0.815979 | 61.4% | `logs/phase2/run-03_focal-minpos_weight-0p1_k-64_00` / `ad61100369524e009e00f973` |
| 2.4 | 0.000000 | 0.498047 | 0.500101 | 44.4% | `logs/phase2/run-04_focal-rank_weight-0p1_00` / `604714bb9ce149e098720440` |
| 2.5 | 0.000000 | 0.498047 | 0.500015 | 70.6% | `logs/phase2/run-05_focal-rank_weight-0p3_00` / `5c6d95a63b794728a6430023` |
| 2.6 | 0.000000 | 0.498047 | 0.500018 | 44.4% | `logs/phase2/run-06_focal-rank_weight-0p1_k-128-128_00` / `bb7c3e3d54284c07afec36bb` |

**Status (2026-07-17):** All six runs completed at 100 epochs (Hydra
`overrides.yaml` in each run directory confirms the planned configuration:
`loss.criterion.gamma=3.0` plus the phase's aux overrides, seed `20260710`).
Every auxiliary configuration degrades the target metric severely.

### Phase 2 conclusion

The auxiliary batch-level terms hinder training rather than help it. The
best aux run (2.3, minpos w=0.1, k=64) reaches TNR 0.124055 — three times
worse than **C\*** alone (0.370795, Phase 1 run 1.3). Four of the six runs
collapse to a degenerate scorer (AUROC ≈ 0.5, TNR = 0): all three rank runs
and minpos w=0.3. The three rank runs end at identical plateau loss values
(`loss_base` 0.164183, `loss_rank` 1.313262) — the same collapsed solution
regardless of weight or k.

The failure mode is the one the inspection criterion above was designed to
catch: the aux terms hijack training. Their weighted share of the total loss
at the end of training is 44–72% in every run — beyond the >50% red line
even at w=0.1 — because `loss_minpos` / `loss_rank` plateau while the focal
base loss decays, so the aux term's relative weight grows as training
converges. This is consistent with the uncontrolled 200-epoch
`loss-ablation` minpos runs, which also ended with near-zero TNR.

Outcome: **C\* + A\* = focal γ=3 with no aux term** (TNR 0.370795). No aux
configuration beats — or even approaches — **C\*** alone.

## Phase 3 — `pos_weight` interaction

Focal/asymmetric focusing and class reweighting overlap: the focusing factors
already suppress easy examples, so the large `auto` ratio (neg/pos of the
train set) may over-boost positives and cost background rejection. Varies
`pos_weight` for the Phase-2 winner — which is **C\*** alone (focal γ=3, no
aux term), so the winner overrides are `loss=focal loss.criterion.gamma=3.0`;
4 runs plus the already-available `auto` result (run 3.4 was added beyond the
originally planned 3-run matrix). A second sweep at seed `20260719` repeats
the four explicit weights. The exploratory `pos_weight=200` extension was
evaluated separately at seeds `20260711` and `20260719`.

| # | Run | Command (append to `TRAIN` + winner overrides) |
| --- | --- | --- |
| 3.0 | auto | (= Phase-1 run 1.3, no new run) |
| 3.1 | pos_weight=1 | `loss.pos_weight=1` |
| 3.2 | pos_weight=10 | `loss.pos_weight=10` |
| 3.3 | pos_weight=50 | `loss.pos_weight=50` |
| 3.4 | pos_weight=100 | `loss.pos_weight=100` |
| 3.5 | pos_weight=200 | `loss.pos_weight=200` |

Original results at seed `20260710` (validation, best checkpoint; `sas.json`
entry `tpr_requested=0.9999`, AUROC from `val.json`):

| # | `tnr_at_tpr_0p9999` | threshold | AUROC | run dir / Aim hash |
| --- | --- | --- | --- | --- |
| 3.0 | 0.370795 | 0.142578 | 0.920179 | `logs/phase1/run-03_focal_gamma-3p0_00` (= run 1.3) |
| 3.1 | 0.340605 | 0.075684 | 0.919496 | `logs/phase3/run-01_pos-weight-1_00` / `fbd73be3dd564f09ad905a7a` |
| 3.2 | 0.371054 | 0.124023 | 0.921029 | `logs/phase3/run-02_pos-weight-10_00` / `27ca51928a0d4f14b4c5a1b9` |
| 3.3 | 0.361573 | 0.158203 | 0.917742 | `logs/phase3/run-03_pos-weight-50_00` / `5f5d956b41c94206a77489ba` |
| 3.4 | **0.390882** | 0.178711 | 0.920833 | `logs/phase3/run-04_pos-weight-100_00` / `83013b82d27b45f9b00832ab` |

The first exploratory `pos_weight=200` run used seed `20260711`:

| # | `tnr_at_tpr_0p9999` | threshold | run dir |
| --- | --- | --- | --- |
| 3.5 | **0.401230** | 0.212891 | `logs/phase3/run-04_pos-weight-200_00` |

Replication results at seed `20260719` (same metric and checkpoint
selection):

| # | `tnr_at_tpr_0p9999` | threshold | run dir |
| --- | --- | --- | --- |
| 3.1 | 0.344438 | 0.079102 | `logs/phase3/run-01_pos-weight-1_01` |
| 3.2 | **0.387238** | 0.121582 | `logs/phase3/run-02_pos-weight-10_01` |
| 3.3 | 0.369905 | 0.151367 | `logs/phase3/run-03_pos-weight-50_01` |
| 3.4 | 0.385186 | 0.179688 | `logs/phase3/run-04_pos-weight-100_01` |
| 3.5 | 0.382388 | 0.194336 | `logs/phase3/run-04_pos-weight-200_01` |

Two-run summaries are shown below. The first `pos_weight=200` run uses seed
`20260711`, not `20260710`, so its mean is not a paired comparison with the
other weights.

| `pos_weight` | first result (seed) | seed `20260719` | mean | spread |
| --- | --- | --- | --- | --- |
| 1 | 0.340605 (`20260710`) | 0.344438 | 0.342522 | 0.003833 |
| 10 | 0.371054 (`20260710`) | 0.387238 | 0.379146 | 0.016184 |
| 50 | 0.361573 (`20260710`) | 0.369905 | 0.365739 | 0.008332 |
| 100 | 0.390882 (`20260710`) | 0.385186 | 0.388034 | 0.005696 |
| 200 | 0.401230 (`20260711`) | 0.382388 | **0.391809** | 0.018842 |

**Status (2026-07-19):** The original four runs, both exploratory
`pos_weight=200` runs, and all seed-`20260719` replicas completed at 100
epochs. Their configs use `loss=focal loss.criterion.gamma=3.0` plus the
listed `loss.pos_weight`. The `auto` reference was not repeated at seed
`20260719`.

### Phase 3 conclusion

Class reweighting still matters on top of focal focusing. Removing it
entirely (`pos_weight=1`, run 3.1) is the worst explicit configuration at
both seeds (TNR 0.340605 and 0.344438), so the focusing factor alone does not
compensate for the class imbalance at the target operating point.

The response is not monotonic in `pos_weight`, and the exact ranking is
seed-sensitive. At seed `20260710`, 100 is best, followed by 10, 50, and 1;
at seed `20260719`, 10 narrowly leads 100, followed by 200, 50, and 1. The
seed-`20260711` `pos_weight=200` run reaches 0.401230, the highest individual
Phase-3 TNR, but its seed-`20260719` replica falls to 0.382388.

`pos_weight=200` consequently has the highest numerical two-run mean
(0.391809), but its runs use a different first seed from the paired sweep and
its spread is the largest (0.018842). `pos_weight=100` has a slightly lower
mean (0.388034), but the highest worst-run TNR (0.385186) and a much smaller
spread (0.005696). The +0.003775 mean advantage for 200 is therefore not
enough to displace 100 without a seed-`20260710` weight-200 run or a broader
common-seed comparison.

The original-seed gain of `pos_weight=100` over `auto` remains +0.020087
(5.4% relative), but `auto` was not replicated, so that margin cannot yet be
called seed-robust. AUROC remains uninformative for the original-seed ranking
(all runs sit at 0.918–0.921).

Outcome: the conservative winner entering Phase 4 remains **focal γ=3 with
`pos_weight=100`** (`loss=focal loss.criterion.gamma=3.0
loss.pos_weight=100`), with two-run mean TNR 0.388034 and worst-run TNR
0.385186. The higher but less stable `pos_weight=200` result remains a
candidate for a common-seed follow-up.

## Phase 4 — Seed robustness and final comparison

The 99.9%-TPR threshold is set by a handful of validation tracks, so
single-run margins can be statistical. The overall winner and BCE baseline
were compared on six common seeds: the existing pinned `20260710` runs plus
five new runs per config at seeds 1–5, launched via `submit/phase4.fish`.
The pinned-seed winner is Phase-3 run 3.4; the pinned-seed BCE reference is
Phase-1 run 1.0.

Results (validation, best checkpoint; `sas.json` entry
`tpr_requested=0.9999`):

| seed | focal γ=3, `pos_weight=100` | threshold | BCE | threshold | paired TNR gain |
| --- | --- | --- | --- | --- | --- |
| `20260710` | 0.390882 | 0.178711 | 0.332205 | 0.001648 | +0.058677 |
| `1` | 0.358400 | 0.168945 | 0.335412 | 0.001282 | +0.022988 |
| `2` | 0.378467 | 0.174805 | 0.362551 | 0.001411 | +0.015916 |
| `3` | 0.378134 | 0.175781 | 0.350283 | 0.001648 | +0.027851 |
| `4` | 0.386156 | 0.177734 | 0.372250 | 0.001411 | +0.013906 |
| `5` | 0.390527 | 0.172852 | 0.353528 | 0.001755 | +0.036999 |

| Config | mean | minimum | maximum | spread |
| --- | --- | --- | --- | --- |
| BCE baseline | 0.351038 | 0.332205 | 0.372250 | 0.040045 |
| focal γ=3, `pos_weight=100` | **0.380428** | **0.358400** | **0.390882** | 0.032482 |

**Status (2026-07-20):** All ten new runs completed at 100 epochs. Focal
beats BCE at every matched seed, with paired absolute gains from 0.013906 to
0.058677. Its mean gain is 0.029389 (8.4% relative).

### Phase 4 conclusion

The paired comparison consistently favors focal γ=3 with
`pos_weight=100`: it beats BCE at all six common seeds, with positive paired
gains of 0.013906–0.058677. This consistent matched-seed result supports the
conclusion that focal is better than BCE under the studied training setup,
so focal γ=3 with `pos_weight=100` is selected as the final model
configuration.

Training stochasticity is nevertheless substantial. Focal's worst result
(0.358400) is below BCE's best (0.372250), and the mean paired gain
(0.029389) is smaller than either config's full seed spread (0.032482 for
focal, 0.040045 for BCE). Thus the predeclared robust-improvement criterion,
which requires separation stronger than the observed seed variation, is not
met. This does not reverse the paired model-selection result; it records
that the improvement does not dominate run-to-run variation and that a
single run is an imprecise performance estimate.

The test set was not used for this decision. It remains reserved for the
final performance report of the selected configuration.

## Final test-set evaluation

For the declared winner only (metric.md procedure):

1. Freeze the score threshold returned by `tnr_at_tpr_0p9999` on validation.
2. Evaluate once on the test set at that fixed threshold; report achieved
   TPR, TNR, FPR, and fraction of tracks sent to extrapolation.
3. Report fixed-threshold performance vs pT, η, pileup; check the
   `loss_pt_0p5_3` / `loss_pt_3_inf` diagnostics for regressions in the
   low-pT bin.

## Bookkeeping

- Actual completed budget: 8 (P1) + 6 (P2) + 10 (P3) + 10 (P4) = 34 full
  runs at 100 epochs each. Phase 4 reuses the pinned-seed focal and BCE runs
  from earlier phases in its six-seed comparison.
- Every run's exact config is stored by Hydra in its run directory; the
  tables above only need the run directory / Aim hash to be reproducible.
- If two configs tie within seed spread, prefer (in order): fewer
  hyperparameters, no aux term, lower γ.

## Conclusions

Focal γ=3 with `pos_weight=100` is the selected configuration from the
controlled ablation. It beats BCE at all six matched Phase-4 seeds and raises
mean validation TNR at TPR ≥ 99.9% from 0.351038 to 0.380428 (+0.029389,
8.4% relative), supporting the conclusion that focal is better than BCE in
this setup. The overlapping seed ranges and the failed predeclared
robust-improvement criterion show that stochastic training variation is
large relative to the improvement; they do not negate the consistent paired
advantage. The untouched test set will be evaluated only for the final
performance report, not for model selection or hyperparameter optimization.

## Reference resolved configuration

The resolved Hydra configuration below is preserved from the representative
Phase-4 focal run at seed 1. It records the complete selected training setup
for reproducing the study and configuring future comparisons. Absolute paths,
device assignment, seed, experiment name, and run name are specific to this
run and should be changed as appropriate when reusing the configuration.

```yaml
model:
  _target_: muonly.nn.LatentCrossAttentionModel
  tracker_track_dim: 7
  dt_segment_dim: 6
  csc_segment_dim: 6
  gem_segment_dim: 6
  rpc_hit_dim: null
  gem_hit_dim: null
  output_dim: 1
  model_dim: 128
  num_heads: 8
  muon_det_latent_len: 64
  muon_det_num_processors: 1
  muon_det_processor_block_weight_sharing: false
  decoder_num_layers: 1
  widening_factor: 4
  dropout: 0
data:
  tracker_track:
    features:
    - px
    - py
    - pz
    - vx
    - vy
    - vz
    - charge
    dim: 7
    preprocessing:
    - _target_: muonly.data.transforms.SignedLog1p
      index:
      - 0
      - 1
      - 2
      - 3
      - 4
      - 5
    - _target_: muonly.data.transforms.MinMaxScaling
      input_min:
      - -14.243
      - -14.302
      - -15.286
      - -4.678
      - -4.946
      - -6.193
      - -1
      input_max:
      - 13.967
      - 12.497
      - 14.909
      - 5.152
      - 3.461
      - 5.638
      - 1
    is_good: true
    target: track_is_trk_muon
  dt_segment:
    features:
    - pos_x
    - pos_y
    - pos_z
    - dir_x
    - dir_y
    - dir_z
    dim: 6
    preprocessing:
    - _target_: muonly.data.transforms.MinMaxScaling
      input_min:
      - -787.279
      - -720.46
      - -652.168
      - -1
      - -1
      - -1
      input_max:
      - 779.118
      - 720.495
      - 652.156
      - 1
      - 1
      - 1
  csc_segment:
    features:
    - pos_x
    - pos_y
    - pos_z
    - dir_x
    - dir_y
    - dir_z
    dim: 6
    preprocessing:
    - _target_: muonly.data.transforms.MinMaxScaling
      input_min:
      - -687.41
      - -686.932
      - -1037.543
      - -1
      - -1
      - -1
      input_max:
      - 685.69
      - 685.244
      - 1037.743
      - 1
      - 1
      - 1
  gem_segment:
    features:
    - pos_x
    - pos_y
    - pos_z
    - dir_x
    - dir_y
    - dir_z
    dim: 6
    preprocessing:
    - _target_: muonly.data.transforms.MinMaxScaling
      input_min:
      - -142.873
      - -144.459
      - -538.79
      - -1
      - -1
      - -1
      input_max:
      - 142.96
      - 144.926
      - 538.79
      - 1
      - 1
      - 1
  rpc_hit: null
  gem_hit: null
data_load:
  train_max_events: null
  val_max_events: null
  test_max_events: null
  batch_size: 128
  eval_batch_size: 128
  num_workers: 0
  pin_memory: false
optim:
  max_epochs: 100
  lr: 0.0003
  weight_decay: 0.01
  beta1: 0.9
  beta2: 0.999
  warmup:
    frac_steps: 0.05
    start_factor: 0.1
  annealing:
    eta_min_factor: 0.01
  max_grad_norm: 1
loss:
  pos_weight: 100
  criterion:
    _target_: muonly.nn.BinaryFocalLoss
    gamma: 3.0
    reduction: none
paths:
  root_dir: /users/hep/slowmoyang/work/muonly/dev-loss
  log_dir: /users/hep/slowmoyang/work/muonly/dev-loss/logs/
  run_dir: /users/hep/slowmoyang/work/muonly/dev-loss/logs//phase4/run-01_focal_gamma-3p0_pos-weight-100_seed-1_00
  work_dir: /users/hep/slowmoyang/work/muonly/dev-loss
  data_dir: /users/hep/joshin/store/muonly/dataset/
  train_file: /users/hep/joshin/store/muonly/dataset//train.h5
  val_file: /users/hep/joshin/store/muonly/dataset//val.h5
  test_file: /users/hep/joshin/store/muonly/dataset//test.h5
torch:
  num_threads: 16
  num_interop_threads: 1
  float32_matmul_precision: high
  device: cuda:0
  precision: bfloat16
  sdpa_backend: math
  seed: 1
  compile: false
  detect_anomaly: false
  check_nan: true
misc:
  tqdm: false
exp: phase4
run: run-01_focal_gamma-3p0_pos-weight-100_seed-1_00
```
