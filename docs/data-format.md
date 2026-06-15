# Data Format

This document describes the tensor shapes and dtypes of the model inputs and output.
It is intended as a reference for exporting trained models to ONNX.

Sources:
- Object definitions and preprocessing: `config/data/mu2030pu.yaml`
- Tensor construction and batching: `src/muonly/data/datasets/tracker_track_selection.py`
- Model `forward()` signature: `src/muonly/nn/models/vanilla_transformer.py`,
  `src/muonly/nn/models/latent_attention.py`
- Active-key selection: `src/muonly/data/utils.py` (`configure_model_in_keys`)

## Overview

The model performs **per-tracker-track binary classification** (target
`track_is_trk_muon`): for each reconstructed tracker track in an event, predict
whether it is a tracker muon.

Each event is a variable-length **set** of objects per type (tracker tracks, muon
detector segments, muon detector hits). A batch packs `B` events by padding every
object type to its per-batch maximum length and attaching a boolean validity mask.

Data flow:

```
HDF5 → TrackerTrackSelectionDataset → preprocessing transforms → collate → model
       (stack features, float32)       (SignedLog1p, MinMaxScaling)   (pad + masks)
```

In `TrackerTrackSelectionDataset`, features are stacked with
`np.stack(axis=1, dtype=np.float32)`, so every feature tensor is **float32**.
`collate` pads with `pad_sequence(batch_first=True, padding_value=0)` and builds a
`{key}_data_mask` for every object key (where `arange(L) < length`).

`B` = batch size. `L_*` = padded sequence length per object type (varies per batch).
`D_*` = feature dimension per object type (fixed).

## Model inputs

The raw model `forward()` consumes **12 tensors**: 6 feature tensors + 6 boolean
masks. The training pipeline wraps the model in a `TensorDictModule` that unpacks
these keys from the batch `TensorDict`; for ONNX export, bypass that wrapper and pass
the 12 tensors directly.

| input | shape | dtype | feature order / meaning |
|---|---|---|---|
| `tracker_track` | `(B, L_trk, 7)` | float32 | `px, py, pz, vx, vy, vz, charge` |
| `tracker_track_data_mask` | `(B, L_trk)` | bool | valid tracker tracks |
| `dt_segment` | `(B, L_dt, 6)` | float32 | `pos_x, pos_y, pos_z, dir_x, dir_y, dir_z` |
| `dt_segment_data_mask` | `(B, L_dt)` | bool | valid DT segments |
| `csc_segment` | `(B, L_csc, 6)` | float32 | `pos_x, pos_y, pos_z, dir_x, dir_y, dir_z` |
| `csc_segment_data_mask` | `(B, L_csc)` | bool | valid CSC segments |
| `gem_segment` | `(B, L_gem_seg, 6)` | float32 | `pos_x, pos_y, pos_z, dir_x, dir_y, dir_z` |
| `gem_segment_data_mask` | `(B, L_gem_seg)` | bool | valid GEM segments |
| `rpc_hit` | `(B, L_rpc, 3)` | float32 | `pos_x, pos_y, pos_z` |
| `rpc_hit_data_mask` | `(B, L_rpc)` | bool | valid RPC hits |
| `gem_hit` | `(B, L_gem_hit, 3)` | float32 | `pos_x, pos_y, pos_z` |
| `gem_hit_data_mask` | `(B, L_gem_hit)` | bool | valid GEM hits |

Notes:
- Mask convention: `True` = real data, `False` = padding. Padding value = `0`.
- `VanillaTransformerModel` requires all 12 inputs.
  `LatentAttentionModel` makes `gem_segment`, `rpc_hit`, `gem_hit` (and their masks)
  optional (`None` default).
- `configure_model_in_keys` (`src/muonly/data/utils.py`) determines which keys are
  active from `config.data.*` flags. `tracker_track` is mandatory; the rest can be
  disabled. Only export the inputs your configuration actually enables.

## Model output

| output | shape | dtype | meaning |
|---|---|---|---|
| `logits` | `(B, L_trk)` | float32 | per-tracker-track **pre-sigmoid** logit |

- Aligned 1:1 with `tracker_track` along dim 1 (one logit per input tracker track).
- Apply `sigmoid` to obtain the muon probability. Training uses `BCEWithLogitsLoss`.
- In `VanillaTransformerModel`, padded positions are zeroed
  (`logits.masked_fill(~tracker_track_data_mask, 0)`); read off only positions where
  `tracker_track_data_mask` is `True`.

## Non-model tensors in the batch

These keys exist in the collated `TensorDict` but are **not** consumed by `forward()`
(labels / aux quantities for loss and evaluation):

| key | shape | dtype | notes |
|---|---|---|---|
| `target` | `(B, L_trk)` | from HDF5 (`track_is_trk_muon`) | ground-truth label per track |
| `tracker_track_pt` | `(B, L_trk)` | from HDF5 (`track_pt`) | track pT, for binned metrics |

`collate` does **not** create a `_data_mask` for `target` or `tracker_track_pt`; reuse
`tracker_track_data_mask` for both.

## Preprocessing

Feature preprocessing happens **inside the Dataset, before the model**
(`config/data/mu2030pu.yaml`):

- `tracker_track`: `SignedLog1p` on indices `[0..5]` (px,py,pz,vx,vy,vz), then
  `MinMaxScaling`.
- `dt/csc/gem_segment`, `rpc/gem_hit`: `MinMaxScaling` only.

The `input_min` / `input_max` constants per object type are in
`config/data/mu2030pu.yaml`. **An ONNX-exported model expects already-preprocessed
inputs** — replicate these transforms in the inference pipeline upstream of the model.
