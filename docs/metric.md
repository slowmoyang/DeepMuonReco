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

An exact specificity-at-sensitivity metric uses every distinct prediction as a
candidate threshold. It must therefore retain all predictions and targets until
the end of validation, making its state grow linearly with the number of tracks.
The validation sample contains many tracks, so this unbounded state can consume
substantial accelerator memory and eventually cause an out-of-memory failure.

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
