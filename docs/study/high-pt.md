# Study: Can the Model Learn High-$p_T$ Tracker Tracks?

Status: **Phase 1 complete — the ceiling gate passes. Phase 2 was attempted
but the specialist run collapsed and is inconclusive; Phase 3 has not been
run.**

## Question

At the global threshold giving 99.99% signal efficiency, background rejection
falls from ~42% near the $p_T$ mode to 5.6% above 50 GeV and 4.0% above
100 GeV. Is this a limit of what the model *can* learn about high-$p_T$ tracks,
or is it a limit of how much high-$p_T$ *training exposure* it receives?

The production constraint is a **single global threshold**: a per-$p_T$
threshold lookup table is not available, so the only path is to make the score
itself stop encoding the $p_T$ prior.

## Scope note

Tracks above 20 GeV are 48,109 of 68.6M validation tracks (0.07% of the
negatives). Lifting their rejection from 6% to 40% removes ~0.02% of
extrapolations. **This study will not visibly move the headline metric or the
CPU budget.** Its value is establishing what the architecture can learn, and
removing a sample-specific shortcut that inverts on non-gun data.

## Baseline

| Item | Value |
| --- | --- |
| Run | `logs/base/260722-132934_transparent-skunk` |
| Config | `no-hit`, `latent_cross_attention`, `loss=focal` ($\gamma=3$, `pos_weight=100`) |
| Epochs / seed | 400 / 1876136271 |
| Validation | 49,000 events, 68,608,145 tracks, 2,509,960 positives |
| Global threshold @ TPR ≥ 99.99% | 0.099609375 |
| Global TNR | 0.4084 |
| `tnr_macro_pt` | **0.2320** |

Scores were dumped once with `scripts/dump-scores.py` and every number below
comes from that table via `scripts/analyze-high-pt.py`; no retraining was
involved.

```bash
uv run python scripts/dump-scores.py \
    -c logs/base/260722-132934_transparent-skunk/checkpoints/best.pt -s val
uv run python scripts/analyze-high-pt.py \
    -i logs/base/260722-132934_transparent-skunk/scores/val.h5
```

## Phase 1 — Diagnosis

All rates are on the validation set. `rej@glob` is at the single global
threshold; `rej@bin` re-optimizes the threshold inside the bin at the stated
per-bin efficiency; `AUC` is the within-bin ROC AUC; `truth μ` is the fraction
of the bin's *background* that is a truth-matched simulated muon; `loss neg` is
the share of the total training loss carried by that bin's negatives under the
run's own focal criterion.

| $p_T$ bin | $n_{bkg}$ | rej@glob | rej@bin 99.9% | rej@bin 99.99% | AUC | truth μ | median bkg score | $P(\text{sig})$ | loss neg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0–1 | 35,208,311 | 0.4182 | 0.5429 | 0.4108 | 0.9249 | 0.000 | 0.193 | 0.026 | 27.67% |
| 1–2 | 21,931,709 | 0.3899 | 0.5293 | 0.4000 | 0.9260 | 0.000 | 0.238 | 0.049 | 23.19% |
| 2–3 | 6,113,614 | 0.4812 | 0.6087 | 0.4858 | 0.9500 | 0.000 | 0.118 | 0.037 | 4.68% |
| 3–4 | 1,734,319 | 0.2779 | 0.4023 | 0.2353 | 0.9245 | 0.000 | 0.311 | 0.036 | 1.63% |
| 4–5 | 554,164 | 0.2807 | 0.4279 | 0.2270 | 0.9184 | 0.000 | 0.307 | 0.035 | 0.52% |
| 5–7 | 325,003 | 0.2844 | 0.4336 | 0.3000 | 0.9184 | 0.000 | 0.301 | 0.034 | 0.31% |
| 7–10 | 114,326 | 0.2497 | 0.4043 | 0.1219 | 0.9192 | 0.000 | 0.348 | 0.050 | 0.133% |
| 10–15 | 50,045 | 0.1996 | 0.3964 | 0.3019 | 0.9122 | 0.002 | 0.438 | 0.113 | 0.089% |
| 15–20 | 18,585 | 0.1481 | 0.4071 | 0.0137 | 0.8944 | 0.004 | 0.547 | 0.239 | 0.052% |
| 20–30 | 16,570 | 0.1116 | 0.3844 | 0.3004 | 0.8736 | 0.009 | 0.641 | 0.396 | 0.067% |
| 30–50 | 12,929 | 0.0783 | 0.3225 | 0.1892 | 0.8456 | 0.017 | 0.723 | 0.618 | 0.074% |
| 50–100 | 9,855 | 0.0561 | 0.2846 | 0.1811 | 0.8235 | 0.033 | 0.770 | 0.833 | 0.068% |
| 100–∞ | 8,755 | 0.0401 | 0.1190 | 0.0426 | 0.8093 | 0.002 | 0.723 | 0.127 | 0.050% |

The per-bin 99.99% column is flagged `low_stat` in every bin above 4 GeV: with
20,896 signal tracks in 30–50 GeV the requirement allows two failures, so the
threshold is set by two tracks. The 15–20 GeV entry (0.0137) is that artifact.
**Judge within-bin ranking on the 99.9% column and on AUC.**

### D1 — Within-bin ranking degrades, but does not collapse

AUC falls smoothly from 0.925–0.950 below 3 GeV to 0.874 / 0.846 / 0.824 in the
20–30 / 30–50 / 50–100 GeV bins. The model still separates high-$p_T$ signal
from high-$p_T$ background well; it is far from chance.

At a matched 99.99% per-bin efficiency, per-bin thresholds recover 2.4–3.2×
more rejection above 20 GeV (0.30 / 0.19 / 0.18 versus 0.11 / 0.08 / 0.06). At
the more stable 99.9% target the recovery is 0.38 / 0.32 / 0.28 versus
0.11 / 0.08 / 0.06. **Most of the deficit is where the global threshold lands
relative to the score distribution, not an inability to rank.**

### D2 — The background is not irreducible

Truth-matched simulated muons make up **0.9% / 1.7% / 3.3%** of the background
in the 20–30 / 30–50 / 50–100 GeV bins. High-$p_T$ background is overwhelmingly
genuine non-muons, not real muons that merely lost arbitration. The label
definition does not cap performance here.

### D3 — The background is not duplicate tracks

For the 48,109 background tracks above 20 GeV, $\Delta R$ to the nearest
positive in the same event has median **0.412**; only **1.9%** lie within
$\Delta R < 0.01$ and 16.7% within 0.05. These are not ghost duplicates of the
gun muon, so the absence of track–track self-attention
(`docs/plan.md` P3.8) is not the binding constraint.

**Gate result: passes.** D2 and D3 together rule out both irreducibility
hypotheses, so retraining is worth the runs.

### D4 — The score is largely the $p_T$ prior

The median *background* score climbs monotonically with $p_T$ — 0.193 at
0–1 GeV to 0.770 at 50–100 GeV — tracking $P(\text{signal} \mid p_T)$, which
climbs 0.026 → 0.833 because the sample is a flat-$p_T$ muon gun on a steeply
falling pileup spectrum. Signal density is flat at ~1000 tracks/GeV above
10 GeV; background density falls from 35M/GeV to 197/GeV. The two cross near
20 GeV.

The global threshold is 0.0996. **Above 50 GeV the median background track
scores roughly 8× above the cut**, so almost nothing is rejected. This is the
mechanism.

The 100–∞ GeV overflow bin is the sharpest evidence: the gun stops at 100 GeV,
so the true prior there is only 0.127, yet the median background score is still
0.723 and rejection is the worst of any bin (0.040). The model has extrapolated
"high $p_T$ ⇒ muon" past the range where that was ever true.

> Note: this overflow bin — 8,755 background and 1,275 signal tracks — was
> silently dropped by the earlier ad-hoc diagnostic, whose binning stopped at
> 100 GeV. The binning is now shared through `config/eval/default.yaml`.

### D5 — Exposure is the starved axis

Negatives above 20 GeV carry **0.259%** of the total training loss; negatives
below 2 GeV carry 50.9%. Positives above 20 GeV carry 0.095%.

The naive count-based estimate is 0.012%, so **focal $\gamma=3$ already
compensates by a factor ~20** — high-$p_T$ negatives are scored badly and are
therefore heavily focused. The residual 0.26% is what the pT weighting has to
close. Recording this because the study was motivated by the count-based
figure, which overstates the starvation.

## Conclusion so far

The high-$p_T$ deficit is a **training-exposure and prior-encoding problem, not
a learnability ceiling**:

1. within-bin ranking is intact (AUC 0.82–0.87);
2. the background is real and learnable (≤3.3% truth muons, ~2% duplicates);
3. the score encodes $P(\text{signal} \mid p_T)$ almost directly, which is what
   puts high-$p_T$ background above a global cut;
4. that prior is a property of the flat-$p_T$ gun, and the model applies it
   beyond 100 GeV where it is false.

## Phase 2 — Ceiling experiment (aborted, inconclusive)

One run. `loss=focal_highpt` restricts the loss support to $p_T \geq 10$ GeV.
Model inputs are unchanged: every track still enters the network and the event
context is identical, only the tracks the loss is computed on change.

```bash
uv run python scripts/train.py --config-name no-hit \
    exp=high-pt run=phase2_specialist \
    loss=focal_highpt torch.seed=20260710
```

Compare against the baseline on within-bin AUC and rejection at a per-bin
99.9% efficiency for $p_T > 20$ GeV. **Do not compare at the global
threshold** — a specialist has no low-$p_T$ calibration, so a single global
operating point is meaningless for it.

- specialist ≫ baseline ⇒ exposure-limited, Phase 3 is the right fix;
- specialist ≈ baseline ⇒ the architecture or the inputs are the ceiling, and
  the answer is `docs/plan.md` P2/P3, not reweighting.

The run at `logs/high-pt/phase2_specialist` was manually stopped during the
epoch 349 validation, after epoch 348 had completed. Validation collapsed to an
all-positive operating point: from epoch 107 onward,
`tnr_at_tpr_0p9999`, `tnr_macro_pt`, and every per-bin TNR remained zero, and
the operating threshold remained zero. Over the same period the validation
loss diverged, reaching 26.70 at epoch 348.

The only saved checkpoint is the global-TNR-selected checkpoint from epoch 6,
with `tnr_at_tpr_0p9999 = 0.0016`. This is not a usable specialist result:
the protocol above requires within-bin AUC and rejection at 99.9% per-bin
efficiency, while checkpointing monitored the global 99.99% operating point
that is explicitly meaningless for a specialist. No converged checkpoint is
available for the intended comparison. **The attempt is therefore an
optimization failure, not evidence for either side of the ceiling gate.**

| Run | AUC 20–30 | AUC 30–50 | AUC 50–100 | rej@bin 99.9% >20 GeV | Aim hash |
| --- | --- | --- | --- | --- | --- |
| baseline | 0.8736 | 0.8456 | 0.8235 | 0.384 / 0.323 / 0.285 | — |
| specialist | — | — | — | **aborted / inconclusive** | — |

## Phase 3 — $p_T$-balanced loss weighting (not yet run)

Five runs sweeping `loss.pt_weight.alpha`, everything else frozen, seed pinned
— the one-knob-per-phase discipline of `docs/study/loss.md`.

```bash
for ALPHA in 0 0.25 0.5 0.75 1.0
    uv run python scripts/train.py --config-name no-hit \
        exp=high-pt run=phase3_alpha-$ALPHA \
        loss=focal_ptbal loss.pt_weight.alpha=$ALPHA torch.seed=20260710
end
```

`alpha` moves the loss share of $p_T > 20$ GeV negatives as follows (count
basis, before focal modulation):

| `alpha` | weight on 50–100 GeV negatives | loss share of $p_T>20$ negatives |
| ---: | ---: | ---: |
| 0 | 1.0 | 0.0124% |
| 0.25 | 6.7 | 0.0766% |
| 0.5 | 39.7 | 0.4251% |
| 0.75 | 185.8 | 1.8537% |
| 1.0 | 558.8 | 5.2128% |

`max_ratio` defaults to 1000 so `alpha=1` is unclamped; at the more obvious
value of 100 every setting above `alpha=0.75` saturates and the top of the
sweep collapses into one experiment.

Judge on `tnr_macro_pt` and the per-bin $p_T > 20$ GeV rejection, **with global
`tnr_at_tpr_0p9999` as the guardrail**: flattening the prior can cost low-$p_T$
rejection, which is where all the statistics and all the CPU are. An `alpha`
that wins `tnr_macro_pt` while losing global TNR is not a win.

**Overfitting watch.** The train set holds roughly 120k background tracks above
20 GeV, upweighted several hundred-fold over 400 epochs. Track train-versus-val
per-bin rejection in those bins explicitly.

| `alpha` | `tnr_at_tpr_0p9999` | `tnr_macro_pt` | rej 20–30 | rej 30–50 | rej 50–100 | rej >100 | Aim hash |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.4084 | 0.2320 | 0.1116 | 0.0783 | 0.0561 | 0.0401 | — |
| 0 | | | | | | | |
| 0.25 | | | | | | | |
| 0.5 | | | | | | | |
| 0.75 | | | | | | | |
| 1.0 | | | | | | | |

An `alpha=0` run at the baseline seed is the reproduction control: the weights
are exactly ones, so it must land on the baseline within seed noise (seed
spread is characterized in `docs/study/loss.md` Phase 4).

## Beyond this study

If Phase 3 recovers high-$p_T$ rejection, the underlying sample bias remains: a
flat-$p_T$ `SingleMu` gun cannot represent the high-$p_T$ prior of real Phase-2
data. Confirming that the model no longer keys on $p_T$ requires a non-gun
PU200 sample, which does not currently exist in
`/users/hep/joshin/store/muonly/dataset/`.
