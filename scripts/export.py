import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import onnx
import onnxruntime as ort

from hydra.utils import instantiate
from omegaconf import OmegaConf
from omegaconf import DictConfig

from rich.logging import RichHandler
from rich_argparse import ArgumentDefaultsRichHelpFormatter

from muonly.utils.onnx import Phase2NoHitModelWrapper

_logger = logging.getLogger(Path(__file__).name)
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()],
)

# The dynamo exporter / onnxscript optimizer emit verbose per-initializer INFO
# logs; quiet them so the export summary stays readable.
for _noisy in ("onnxscript", "onnx_ir", "torch.onnx"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

# Wrapper forward() argument order. Feature tensors and their masks are paired;
# downstream consumers (e.g. CMSSW ONNXRuntime) must feed inputs in this order.
INPUT_NAMES = [
    "tracker_track",
    "tracker_track_data_mask",
    "dt_segment",
    "dt_segment_data_mask",
    "csc_segment",
    "csc_segment_data_mask",
    "gem_segment",
    "gem_segment_data_mask",
]
OUTPUT_NAMES = ["logits"]


def build_preprocessing(obj_config: DictConfig) -> nn.Sequential:
    """Build a per-object preprocessing module from its config.

    ``configure_preprocessing`` wraps transforms in ``Compose`` (a plain class),
    whose buffers would NOT register as submodule buffers and therefore would not
    be baked into the ONNX graph. Use ``nn.Sequential`` instead so the
    ``MinMaxScaling`` min/max buffers export as graph constants.
    """
    return nn.Sequential(*instantiate(obj_config["preprocessing"]))


def make_dummy_inputs() -> tuple[torch.Tensor, ...]:
    """Synthetic raw (pre-preprocessing) inputs matching the wrapper signature.

    Masks are all-True: a fully-masked object type yields a softmax over an empty
    set inside the attention encoders, producing NaNs.
    """
    # batch_size must be >= 2: torch.export 0/1-specializes size-1 dims, which
    # would freeze the dynamic "batch" axis to a static 1 in the exported graph.
    batch_size = 2
    len_trk, len_dt, len_csc, len_gem = 500, 20, 30, 40

    def feats(length: int, dim: int) -> torch.Tensor:
        return torch.randn(batch_size, length, dim, dtype=torch.float32)

    def mask(length: int) -> torch.Tensor:
        return torch.ones(batch_size, length, dtype=torch.bool)

    return (
        feats(len_trk, 7),
        mask(len_trk),
        feats(len_dt, 6),
        mask(len_dt),
        feats(len_csc, 6),
        mask(len_csc),
        feats(len_gem, 6),
        mask(len_gem),
    )


def build_dynamic_shapes() -> dict[str, dict[int, "torch.export.Dim"]]:
    """Dynamic axes: batch shared by all inputs; independent seq length per type.

    Each feature tensor shares its sequence-length dim with its own mask.
    """
    Dim = torch.export.Dim
    batch = Dim("batch")
    len_trk = Dim("len_trk")
    len_dt = Dim("len_dt")
    len_csc = Dim("len_csc")
    len_gem = Dim("len_gem")
    return {
        "tracker_track": {0: batch, 1: len_trk},
        "tracker_track_data_mask": {0: batch, 1: len_trk},
        "dt_segment": {0: batch, 1: len_dt},
        "dt_segment_data_mask": {0: batch, 1: len_dt},
        "csc_segment": {0: batch, 1: len_csc},
        "csc_segment_data_mask": {0: batch, 1: len_csc},
        "gem_segment": {0: batch, 1: len_gem},
        "gem_segment_data_mask": {0: batch, 1: len_gem},
    }


def verify(
    output_path: Path,
    wrapper: nn.Module,
    dummy_inputs: tuple[torch.Tensor, ...],
) -> None:
    """Check the ONNX graph and compare ONNXRuntime output against PyTorch."""
    onnx.checker.check_model(onnx.load(str(output_path)))
    _logger.info("onnx.checker.check_model passed")

    with torch.inference_mode():
        torch_logits = wrapper(*dummy_inputs).cpu().numpy()

    session = ort.InferenceSession(
        str(output_path), providers=["CPUExecutionProvider"]
    )
    ort_feeds = {
        name: tensor.cpu().numpy()
        for name, tensor in zip(INPUT_NAMES, dummy_inputs)
    }
    ort_logits = np.asarray(session.run(OUTPUT_NAMES, ort_feeds)[0])

    max_abs_diff = float(np.abs(torch_logits - ort_logits).max())
    _logger.info(f"max abs diff (torch vs onnxruntime): {max_abs_diff:.3e}")
    if not np.allclose(torch_logits, ort_logits, atol=1e-4):
        raise RuntimeError(
            f"ONNX output mismatch: max abs diff {max_abs_diff:.3e} exceeds 1e-4"
        )
    _logger.info("Numerical equivalence verified (atol=1e-4)")


def run(
    ckpt_path: Path,
    output_path: Path | None,
    opset: int,
    args_list: list[str],
) -> None:
    run_dir = ckpt_path.parents[1]
    if output_path is None:
        output_path = run_dir / "model.onnx"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    base_config = OmegaConf.load(run_dir / "config.yaml")
    cli_config = OmegaConf.from_cli(args_list)
    config = OmegaConf.merge(base_config, cli_config)
    if not isinstance(config, DictConfig):
        raise ValueError("Expected config to be a DictConfig")

    # ------------------------------------------------------------------------
    # Model (export on CPU)
    # ------------------------------------------------------------------------
    model = instantiate(config.model)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    del checkpoint
    model = model.eval()
    _logger.info(f"Loaded checkpoint from {ckpt_path}")

    # ------------------------------------------------------------------------
    # Wrap model with baked-in preprocessing
    # ------------------------------------------------------------------------
    wrapper = Phase2NoHitModelWrapper(
        model=model,
        tracker_track_preprocessing=build_preprocessing(config.data.tracker_track),
        dt_segment_preprocessing=build_preprocessing(config.data.dt_segment),
        csc_segment_preprocessing=build_preprocessing(config.data.csc_segment),
        gem_segment_preprocessing=build_preprocessing(config.data.gem_segment),
    ).eval()

    # ------------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------------
    dummy_inputs = make_dummy_inputs()

    # NOTE: dynamo exporter (default in torch >= 2.9). If it ever chokes on the
    # in-place index-put inside transforms.Indexible.forward, fall back to
    # dynamo=False with a `dynamic_axes` dict (the legacy tracer handles it).
    torch.onnx.export(
        wrapper,
        dummy_inputs,
        str(output_path),
        input_names=INPUT_NAMES,
        output_names=OUTPUT_NAMES,
        dynamic_shapes=build_dynamic_shapes(),
        opset_version=opset,
        dynamo=True,
    )
    _logger.info(f"Exported ONNX model to {output_path}")

    # ------------------------------------------------------------------------
    # Verify
    # ------------------------------------------------------------------------
    verify(output_path=output_path, wrapper=wrapper, dummy_inputs=dummy_inputs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a trained model to ONNX with baked-in preprocessing.",
        formatter_class=ArgumentDefaultsRichHelpFormatter,
    )
    parser.add_argument(
        "-c",
        "--ckpt",
        dest="ckpt_path",
        type=Path,
        required=True,
        help="Path to the model checkpoint (e.g., 'runs/exp1/checkpoints/best.pt')",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_path",
        type=Path,
        default=None,
        help="Output .onnx path (default: '<run_dir>/model.onnx')",
    )
    parser.add_argument(
        "--opset",
        dest="opset",
        type=int,
        default=18,
        help="ONNX opset version",
    )
    args, unknown_args = parser.parse_known_args()

    run(**vars(args), args_list=unknown_args)


if __name__ == "__main__":
    main()
