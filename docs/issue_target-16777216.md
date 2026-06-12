# Invalid Target Value `16777216` During Validation

## Summary

A training run failed during initial validation because TorchMetrics received
an invalid binary-classification target:

```text
RuntimeError: Detected the following values in `target`:
tensor([       0,        1, 16777216])
but expected only the following values [0, 1].
```

The invalid value was not present in the source HDF5 dataset. Diagnostics
indicate that the target tensor was corrupted during the CUDA mixed-precision
execution path. The most likely source is an optimized
scaled-dot-product-attention (SDPA) CUDA kernel, but this has not been proven.

The immediate workarounds are to run in full `float32` precision or force the
math SDPA backend when using `bfloat16`.

## Incident

The failing command was:

```bash
uv run python scripts/train.py \
  torch.device=cuda:1 \
  optim.max_epochs=20 \
  model.encoder_num_layers=1 \
  model.decoder_num_layers=4 \
  loss=pt_bin
```

The canonical log is:

```text
logs/no-hit/260612-154832_sensible-cheetah/hydra.log
```

The run failed during epoch-zero validation in
`BinarySpecificityAtSensitivity.update()`. The invalid value became visible
after the masked target was converted from `float` back to `long` for metric
evaluation:

```python
target = batch["target"][mask].float()
# model inference and BCE evaluation
target = target.long().cpu()
```

The failure was not raised by the momentum-bin-balanced loss. Validation uses
the same target and tracker-track mask regardless of `loss=pt_bin`.

## Environment

The incident and follow-up diagnostics used:

| Component | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5090 |
| GPU compute capability | 12.0 |
| PyTorch | `2.10.0+cu128` |
| CUDA runtime reported by PyTorch | 12.8 |
| cuDNN | 91002 |
| TensorDict | 0.11.0 |
| TorchMetrics | 1.8.2 |
| Training precision | `bfloat16` autocast |
| `torch.compile` | Disabled |
| SDPA backends enabled | Flash, memory-efficient, and math |

## Investigation

### Dataset Targets

All selected validation targets were scanned directly from:

```text
/users/hep/joshin/store/muonly/dataset/val.h5
```

The scan applied `track_is_good_track`, matching the dataset loader's
selection. Across all 49,000 validation events and 68,608,145 selected tracks,
the only target values were:

```text
[0, 1]
```

No invalid target was found. This rules out the HDF5 labels and the good-track
selection as the source of `16777216`.

### Loading, Padding, and Masking

The dataset loader reads `track_is_trk_muon`, applies the same good-track mask
used for tracker-track features and `track_pt`, and pads targets with zero
during collation. Validation then selects targets with
`tracker_track_data_mask`.

These operations preserve binary targets and do not expose padded values to
the metrics. No evidence of corruption was found in this path.

### CUDA Execution

An isolated `bfloat16` CUDA diagnostic checked targets:

1. Before transfer to the GPU.
2. After transfer to the GPU.
3. After model inference.
4. After BCE evaluation.

Targets remained exactly `0` or `1` for five batches. The process then exited
with segmentation fault status `139`. A segmentation fault after apparently
valid CUDA results is consistent with an asynchronous kernel fault or memory
corruption.

Two comparison runs over the same five validation batches completed without
target corruption or a segmentation fault:

- Full `float32` execution.
- `bfloat16` autocast with the math-only SDPA backend.

## Interpretation

`16777216` is `2^24`, but its appearance is not consistent with ordinary
floating-point rounding. Converting the valid binary values `0` and `1`
between integer, `float32`, and `bfloat16` preserves them exactly. Mixed
precision should also not modify the separate target tensor.

The available evidence instead points to CUDA memory corruption during the
optimized mixed-precision attention path:

- The source labels are valid.
- The CPU loading and masking path is valid.
- Full precision completes cleanly.
- Math-only SDPA with `bfloat16` completes cleanly.
- The default optimized SDPA path produced both an invalid target and a later
  segmentation fault.

The likely source is an optimized SDPA kernel used by
`torch.nn.functional.scaled_dot_product_attention` on the RTX 5090 with the
installed PyTorch/CUDA stack. This remains a hypothesis until a specific
backend and minimal reproducer reliably trigger the corruption.

## Workarounds

### Use Full Precision

The simplest workaround is:

```bash
uv run python scripts/train.py \
  torch.device=cuda:1 \
  torch.precision=float32 \
  optim.max_epochs=20 \
  model.encoder_num_layers=1 \
  model.decoder_num_layers=4 \
  loss=pt_bin
```

This avoids the failing mixed-precision path, at the cost of increased memory
use and reduced throughput.

### Force the Math SDPA Backend

To retain `bfloat16`, run model inference and training under a math-only SDPA
context:

```python
from torch.nn.attention import SDPBackend, sdpa_kernel

with sdpa_kernel(SDPBackend.MATH):
    # training and validation
    ...
```

This completed the diagnostic successfully, but is expected to be slower than
optimized attention backends.

## Recommended Follow-Up

1. Reproduce the failure separately with flash SDPA and memory-efficient SDPA
   to identify the affected backend.
2. Run with `CUDA_LAUNCH_BLOCKING=1` and CUDA memory-checking tools to locate
   the first failing kernel.
3. Reduce the model and batch shapes to a standalone SDPA reproducer.
4. Test other supported PyTorch and CUDA builds on the RTX 5090.
5. Add an explicit binary-target assertion before loss and metric updates so
   future corruption fails near its first observable occurrence.
6. Report the minimal reproducer upstream if the issue remains specific to an
   optimized PyTorch SDPA backend.

## Conclusion

The invalid target value was not caused by the dataset, `pt_bin` balancing, or
normal floating-point precision behavior. Current evidence supports a CUDA
mixed-precision attention kernel or memory-corruption issue. Until the affected
backend is isolated, use full `float32` precision or force the math SDPA
backend.
