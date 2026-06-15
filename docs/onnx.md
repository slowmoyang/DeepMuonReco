# ONNX Export

This document describes how a trained model is exported to ONNX with
**preprocessing baked into the graph**, so downstream consumers (e.g. CMSSW
ONNXRuntime) feed **raw** features and read logits directly.

Sources:
- Export script: `scripts/export.py`
- Model wrapper: `src/muonly/utils/onnx.py` (`Phase2NoHitModelWrapper`)
- Input/output tensor reference: `docs/data-format.md`

## Why a wrapper

The bare model expects **already-preprocessed** inputs — the training pipeline
applies `SignedLog1p` / `MinMaxScaling` inside the `Dataset`, before the model
(`docs/data-format.md`). If exported alone, every consumer would have to replicate
those transforms upstream.

`Phase2NoHitModelWrapper` (`src/muonly/utils/onnx.py`) wraps the model together
with the per-object preprocessing modules. Its `forward()` runs preprocessing then
model inference, so the exported `.onnx` accepts **raw** features and the transforms
become graph constants.

The wrapper targets the **no-hit** configuration: `rpc_hit` and `gem_hit` disabled
(`config.data.rpc_hit = null`, `gem_hit = null`). It accepts 8 inputs and passes
`rpc_hit=gem_hit=None` to the underlying model.

## Inputs and output

8 inputs (4 raw feature tensors + 4 boolean masks), in `forward()` order:

| input | shape | dtype |
|---|---|---|
| `tracker_track` | `(B, L_trk, 7)` | float32 (raw) |
| `tracker_track_data_mask` | `(B, L_trk)` | bool |
| `dt_segment` | `(B, L_dt, 6)` | float32 (raw) |
| `dt_segment_data_mask` | `(B, L_dt)` | bool |
| `csc_segment` | `(B, L_csc, 6)` | float32 (raw) |
| `csc_segment_data_mask` | `(B, L_csc)` | bool |
| `gem_segment` | `(B, L_gem, 6)` | float32 (raw) |
| `gem_segment_data_mask` | `(B, L_gem)` | bool |

Output: `logits (B, L_trk)`, float32, **pre-sigmoid**. Apply `sigmoid` and read
only positions where `tracker_track_data_mask` is `True`. (`LatentAttentionModel`
does not zero padded positions — see `docs/data-format.md`.)

Feature order per object type and raw value ranges: see `docs/data-format.md` and
the run's `config.yaml` (`data.<object>.features` / `preprocessing`).

## How `scripts/export.py` works

1. Load the run's `config.yaml` and checkpoint (`["model"]` state dict) on CPU,
   mirroring `scripts/predict.py`.
2. Build per-object preprocessing as `nn.Sequential(*instantiate(cfg.preprocessing))`.
   - **Not** `configure_preprocessing` / `Compose`: `Compose` is a plain class, so
     its transform buffers (`MinMaxScaling` min/max) would not register as submodule
     buffers and would not bake into the graph. `nn.Sequential` of the `nn.Module`
     transforms does.
3. Wrap in `Phase2NoHitModelWrapper`.
4. `torch.onnx.export(..., dynamo=True, opset_version=18)` with the 8 named inputs,
   `logits` output, and dynamic axes: batch `B` plus an independent sequence length
   per object type (each feature tensor shares its `L` with its mask).
5. Verify: `onnx.checker.check_model`, then compare ONNXRuntime vs PyTorch on
   synthetic random inputs (all-True masks) — asserts max abs diff `< 1e-4`.

### Dynamo dependency

The dynamo exporter (`dynamo=True`, default in torch ≥ 2.9) requires
**`onnxscript`** (`pyproject.toml`). Without it, export fails with
`ModuleNotFoundError: No module named 'onnxscript'`.

If dynamo ever chokes on the in-place index-put inside `transforms.Indexible.forward`
(`output[..., index] = ...`), fall back to `dynamo=False` with a `dynamic_axes`
dict — the legacy TorchScript tracer handles index_put.

### Masks

Masks are bool, `True` = real data. The synthetic verification uses **all-True**
masks: a fully-masked object type produces a softmax over an empty set inside the
attention encoders, yielding NaNs.

## Usage

```bash
python scripts/export.py -c logs/<exp>/<run>/checkpoints/best.pt
```

- `-c/--ckpt` (required): checkpoint path.
- `-o/--output` (default `<run_dir>/model.onnx`): output `.onnx` path.
- `--opset` (default 18): ONNX opset version.
- Extra `key=value` args pass through to OmegaConf CLI override (as in `predict.py`).

Output: `<run_dir>/model.onnx`, with `onnx.checker` passing and a logged max abs diff
(torch vs ONNXRuntime) `< 1e-4`.
