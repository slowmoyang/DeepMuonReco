# Deep-Learning Preselection for Tracker Muon Reconstruction

## Overview

This project investigates whether a deep-learning model can reduce the cost of
CMS tracker muon reconstruction in the Phase-2 environment. The model assigns a
score to each eligible tracker track before the expensive track-to-muon-system
extrapolation. Only tracks passing a configurable score threshold would enter
the standard extrapolation, matching, and arbitration workflow.

The objective is to reduce the number of tracks extrapolated by
`MuonIdProducer`, and therefore its reconstruction time, while preserving
tracker-muon efficiency. The project does not replace the tracker-muon
definition or the downstream reconstruction logic.

## Motivation

Tracker muon reconstruction starts from inner tracker tracks and tests whether
they are compatible with measurements in the muon system. This requires
propagating each eligible track through the detector geometry and magnetic
field. The extrapolation and subsequent matching are computationally
expensive.

The number of pileup interactions, and consequently the number of reconstructed
tracker tracks, will increase in Phase-2. Extrapolating every eligible track
therefore risks increasing the tracker-muon reconstruction time substantially.
Most tracker tracks do not become tracker muons, so a sufficiently efficient
and inexpensive preselection could avoid unnecessary extrapolations.

## Existing Tracker Muon Reconstruction

`MuonIdProducer` currently:

1. Applies kinematic and quality requirements to inner tracker tracks.
2. Builds a temporary `reco::Muon` for each accepted track.
3. Extrapolates the track to the muon system.
4. Fills compatible DT, CSC, GEM, and other chamber matches.
5. Sets the `TrackerMuon` type bit when the candidate has enough matches.
6. Arbitrates shared segments and may remove the tracker-muon classification.

The final post-arbitration `TrackerMuon` bit defines the target used by this
project. See [MuonIdProducer: Tracker Muon Identification
Logic](muon-id-producer.md) for a detailed description of the existing
algorithm.

## Proposed Reconstruction Flow

The proposed production workflow inserts model inference after the existing
good-track selection and before track extrapolation:

1. Apply the standard `MuonIdProducer` good-track selection.
2. Evaluate all eligible tracks with the preselection model.
3. Extrapolate only tracks whose score exceeds a configurable threshold.
4. Run the standard matching, tracker-muon identification, and arbitration on
   the selected tracks.

Muon-system segments and hits are assumed to have already been reconstructed
and remain available as model inputs. The model is therefore intended to
replace unnecessary track extrapolation and matching, not muon-detector local
reconstruction.

The score threshold controls the tradeoff between tracker-muon efficiency and
the number of tracks sent to extrapolation. A production threshold has not yet
been selected.

## Supervision and Data

The task is per-track binary classification within an event:

- **Positive:** an eligible tracker track associated with a final
  post-arbitration muon whose `TrackerMuon` bit is set.
- **Negative:** an eligible tracker track that does not satisfy that
  definition.

The current dataset uses `track_is_good_track` to reproduce the initial
good-track mask and `track_is_trk_muon` as the classification target. Event data
are converted from ROOT ntuples to HDF5 for training and inference.

The current prototype uses the following inputs:

| Object | Features |
| --- | --- |
| Tracker track | `px`, `py`, `pz`, `vx`, `vy`, `vz`, charge |
| DT, CSC, and GEM segment | Global position and direction |
| RPC and GEM hit | Global position |

Tracker-track momentum and vertex coordinates receive a signed logarithmic
transform followed by min-max scaling. Muon-system measurements receive
min-max scaling. Variable-length object collections are padded and masked
during batching.

## Current Model Prototypes

The repository contains two event-level model families. Both produce one logit,
and therefore one classification score, for every eligible tracker track.

- **Vanilla transformer:** a baseline in which tracker-track embeddings attend
  to the combined muon-system measurements.
- **Latent-attention model:** a more scalable prototype that compresses tracker
  tracks and muon-system measurements into latent representations before
  producing per-track scores.

The current training workflow supports binary cross-entropy, automatic positive
class weighting, and momentum-bin loss balancing. Validation currently records
losses, score distributions, AUROC, and TNR at a minimum TPR of 99.9% in
selected track-momentum regions. `scripts/predict.py` writes per-event,
per-track scores to HDF5 for downstream studies.

These implementations are research prototypes. No architecture has been
selected for production use.

## Evaluation

Model quality alone is insufficient: the final decision must account for both
physics performance and reconstruction cost. The primary model-comparison
metric is the maximum true negative rate (TNR) at a measured true positive rate
(TPR) of at least 99.9%, computed globally over all eligible tracks. This
prioritizes preserving tracker-muon efficiency while measuring how many
negative tracks can avoid extrapolation. The corresponding validation-set score
threshold is a candidate global operating threshold.

Current validation also computes independently optimized TNR values in
track-momentum regions. These are diagnostics and do not describe performance
at one global threshold. See [Primary Evaluation Metric](metric.md) for metric
definitions, TorchMetrics behavior, and the evaluation procedure.

Evaluation should additionally scan or apply the selected score threshold and
measure:

- Tracker-muon efficiency and efficiency loss relative to the unfiltered
  `MuonIdProducer` workflow.
- TNR, false positive rate, and the fraction of eligible tracks sent to
  extrapolation.
- Fixed-threshold efficiency and TNR versus track transverse momentum,
  pseudorapidity, and event pileup.
- Model inference latency, memory use, and scaling with event occupancy.
- `MuonIdProducer` timing with and without preselection.
- Net end-to-end reconstruction speedup after including inference cost.

TNR at a minimum TPR of 99.9% is the primary classification metric for comparing
models, but it does not directly measure timing reduction. The final production
threshold and model must be selected using both physics performance and
end-to-end timing, not classification metrics such as AUROC alone.

## Current Status and Future Work

The repository currently provides dataset loading and preprocessing, model
prototypes, training, validation, checkpointing, and prediction workflows. The
remaining work required to demonstrate the project objective includes:

1. Validate labels and physics performance on representative Phase-2 samples.
2. Produce threshold scans and differential efficiency/rejection studies.
3. Measure standalone inference cost and compare candidate architectures.
4. Integrate inference before tracker-track extrapolation in CMSSW.
5. Measure end-to-end `MuonIdProducer` timing and select a production operating
   point.

At present, the project has no CMSSW inference integration, selected production
threshold, final model architecture, or measured end-to-end reconstruction
speedup.
