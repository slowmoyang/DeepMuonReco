# Primary Evaluation Metric

## Operating Objective

The preselection model should avoid unnecessary tracker-track extrapolation
while preserving tracks that would become final tracker muons. Its primary
model-comparison metric is therefore the **true negative rate (TNR) at a
minimum true positive rate (TPR) of 99.9%**, evaluated over all eligible
tracker tracks.

For this metric:

- A **positive** is an eligible tracker track associated with a final
  post-arbitration muon whose `TrackerMuon` bit is set.
- A **negative** is any other eligible tracker track.
- A predicted positive passes the model score threshold and is sent to the
  existing extrapolation, matching, and arbitration workflow.

The confusion-matrix rates are:

```text
TPR = TP / (TP + FN)
TNR = TN / (TN + FP)
FPR = FP / (TN + FP) = 1 - TNR
```

TPR is the signal efficiency: the fraction of tracker-muon tracks retained by
the preselection. TNR, also called specificity or the background rejection
fraction, is the fraction of negative tracks rejected before extrapolation.
This project maximizes TNR subject to `TPR >= 99.9%` because losing tracker
muons conflicts with the reconstruction goal, while rejecting more negative
tracks increases the opportunity to reduce reconstruction cost.

TNR should not be confused with the background rejection factor, `1 / FPR`.

## TorchMetrics Definition

The validation metric can be computed with:

```python
BinarySpecificityAtSensitivity(
    min_sensitivity=0.999,
    thresholds=100_001,
)
```

Here, sensitivity is TPR and specificity is TNR. TorchMetrics constructs an ROC
curve and returns:

1. The maximum measured TNR among score thresholds whose measured TPR is at
   least 99.9%.
2. The corresponding score threshold.

The returned value is therefore TNR at **TPR greater than or equal to 99.9%**,
not necessarily TNR at exactly 99.9% TPR.

### Why Validation Uses Histograms

The histogram-based implementation was introduced after the exact TorchMetrics
calculation failed on a validation sample containing 68,608,145 tracks. The
failure occurred while `_binary_clf_curve` was sorting and reindexing the stored
predictions:

```text
File ".../torchmetrics/functional/classification/precision_recall_curve.py", line 62, in _binary_clf_curve
    preds = preds[desc_score_indices]
            ~~~~~^^^^^^^^^^^^^^^^^^^^
IndexError: index 144115188090131662 is out of bounds for dimension 0 with size 68608145
```

The exact calculation retains every prediction and target until the end of
validation, then sorts the full prediction array to use every distinct score as
a candidate threshold. At this sample size, that exact sorting and reindexing
path produced the error above. Avoiding this large raw-state operation was the
original reason for replacing it with a histogram-based calculation.

The histogram-based metric instead accumulates positive and negative score
counts in 100,001 uniformly spaced bins over `[0, 1]`. At the end of validation,
reverse cumulative sums over these histograms give the true-positive and
false-positive counts at every threshold. Its state is fixed by the number of
bins rather than the number of evaluated tracks, so memory remains bounded
throughout validation and histogram states can be summed across workers.

This bounded-memory behavior comes with a controlled approximation: thresholds
have a score-grid spacing of `1e-5`, so the reported TNR and operating threshold
are quantized to that resolution. This tradeoff is appropriate for repeated
validation during training; final fixed-threshold evaluation still reports the
rates measured directly on the independent evaluation sample.

The headline metric should be computed once over all tracks that pass the
standard `MuonIdProducer` good-track selection. Its returned threshold is a
candidate global operating threshold. Metrics optimized independently in
different transverse-momentum regions are useful diagnostics, but their TNR
values generally correspond to different thresholds and do not describe the
performance of a single global operating point.

## Secondary Metric: `tnr_macro_pt`

The headline TNR is an average over negatives, and 99.9% of the negatives sit
below 20 GeV. It is therefore almost blind to the high-pT region, where
rejection is worst — a model can lose or gain everything above 20 GeV without
moving `tnr_at_tpr_0p9999` by a measurable amount.

`PtBinnedOperatingPoint` (`muonly.nn.metrics`) reports, at the **same single
global threshold** the headline metric returns, the TNR of each pT bin and
their unweighted mean, `tnr_macro_pt`. Because every bin counts equally
regardless of population, the sparse high-pT bins become visible.

- The pT binning comes from `config/eval/default.yaml` and is shared with the
  pT-balanced loss weighting and the evaluation tables.
- The metric keeps the bounded-memory histogram design described above; its
  state is `(n_pt_bins, 2, 100_001)` and is fixed by the binning, not by the
  number of evaluated tracks.
- The per-bin rates are **not** independently optimized thresholds. They
  describe the differential behaviour of one global cut, which is what the
  production configuration will actually apply.

`tnr_macro_pt` is a secondary metric. `tnr_at_tpr_0p9999` remains the primary
model-selection criterion and the guardrail: a change that improves
`tnr_macro_pt` while degrading the global TNR has traded away rejection in the
bins that carry the reconstruction cost. Report both.

Best-checkpoint selection is configured by `eval.monitor` and defaults to
`tnr_at_tpr_0p9999` (max). Selecting on validation loss is not valid across
runs whose criterion or loss weighting differs.

## Evaluation Procedure

Use the metric as follows:

1. Compare candidate models using global validation-set TNR at
   `TPR >= 99.9%`.
2. Select a candidate global score threshold from the validation sample and
   freeze it before final evaluation.
3. Apply that fixed threshold to an independent test sample.
4. Report the achieved TPR, TNR, false positive rate, and fraction of eligible
   tracks sent to extrapolation.
5. Report the same fixed-threshold performance versus track transverse
   momentum, pseudorapidity, and event pileup.
6. Measure inference cost, `MuonIdProducer` timing, and end-to-end
   reconstruction speedup at the fixed threshold.

AUROC, score distributions, and independently optimized per-region TNR values
remain useful diagnostics, but they are not the primary model-selection
criterion.

## Interpretation and Limitations

TNR is the fraction of negative tracks avoided, not the fraction of all
eligible tracks avoided. The total fraction sent to extrapolation also depends
on the positive and negative class prevalence. TNR likewise does not directly
measure timing reduction: the cost of model inference, the cost distribution
of accepted tracks, and downstream reconstruction behavior must be measured.

An operating point at 99.9% TPR is sensitive to validation-sample statistics.
The sample must contain enough positive tracks to estimate a 0.1% inefficiency
reliably, and performance should be checked on statistically adequate
differential samples. The final production threshold must be validated on
representative Phase-2 data and selected using both physics performance and
end-to-end timing.
