import argparse
import json
import logging
import os
import platform
import subprocess
import time
from datetime import datetime
from pathlib import Path
import warnings

import h5py as h5
import numpy as np
import torch
from torch.utils.data import DataLoader

import onnxruntime as ort

from omegaconf import OmegaConf
from omegaconf import DictConfig

import tqdm.rich
from tqdm import TqdmExperimentalWarning
from rich.logging import RichHandler
from rich_argparse import ArgumentDefaultsRichHelpFormatter

from muonly.data.datasets import TrackerTrackSelectionDataset

_logger = logging.getLogger(Path(__file__).name)
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()],
)

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

# ONNX input feed order, matching INPUT_NAMES in scripts/export.py. The exported
# graph bakes in preprocessing, so these are RAW (pre-preprocessing) features
# paired with their boolean masks. No-hit config: rpc_hit / gem_hit excluded.
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

# Object collections to count per event (one mask each, True = real object).
OBJECT_NAMES = [
    "tracker_track",
    "dt_segment",
    "csc_segment",
    "gem_segment",
]


def collect_cpu_info() -> dict:
    """Best-effort CPU / hardware description for reproducibility."""
    info: dict = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "os_cpu_count": os.cpu_count(),
        # Cores this process is actually allowed to use (cgroup / Slurm aware).
        "allowed_cpu_count": len(os.sched_getaffinity(0)),
    }

    # Parse /proc/cpuinfo for the CPU model name and logical core count.
    try:
        with open("/proc/cpuinfo") as cpuinfo:
            lines = cpuinfo.readlines()
        model_names = {
            line.split(":", 1)[1].strip()
            for line in lines
            if line.startswith("model name")
        }
        info["model_name"] = sorted(model_names)
        info["logical_cores"] = sum(1 for line in lines if line.startswith("processor"))
    except OSError:
        pass

    # Optional richer dump; ignored if lscpu is unavailable.
    try:
        info["lscpu"] = subprocess.check_output(
            ["lscpu"], text=True, stderr=subprocess.DEVNULL
        )
    except (OSError, subprocess.SubprocessError):
        pass

    return info


def summarize_latency(latency_ns: np.ndarray) -> dict:
    """Summary statistics (ns) over recorded per-event latencies."""
    if len(latency_ns) == 0:
        return {}
    return {
        "count": int(len(latency_ns)),
        "min_ns": int(latency_ns.min()),
        "median_ns": float(np.median(latency_ns)),
        "mean_ns": float(latency_ns.mean()),
        "p95_ns": float(np.percentile(latency_ns, 95)),
        "p99_ns": float(np.percentile(latency_ns, 99)),
        "max_ns": int(latency_ns.max()),
    }


def run(
    model_path: Path,
    split: str,
    threads: int,
    warmup: int,
    max_events: int | None,
    args_list: list[str],
) -> None:
    """Measure per-event ONNX inference latency at batch size 1."""
    run_dir = model_path.parent
    output_dir = run_dir / "benchmark"
    output_dir.mkdir(exist_ok=True)
    h5_path = output_dir / f"{split}-onnx-latency.h5"
    json_path = output_dir / f"{split}-onnx-latency.json"

    base_config = OmegaConf.load(run_dir / "config.yaml")
    cli_config = OmegaConf.from_cli(args_list)
    config = OmegaConf.merge(base_config, cli_config)
    if not isinstance(config, DictConfig):
        raise ValueError("Expected config to be a DictConfig")

    # ---------------------------------------------------------------------------
    # ONNXRuntime session
    #
    # By default ONNXRuntime spawns one thread per CPU core. Cap the op thread
    # pools so the job stays within its allotted CPU budget (e.g. on a shared
    # node or under a Slurm cpus-per-task limit). Torch is similarly capped.
    # ---------------------------------------------------------------------------
    torch.set_num_threads(threads)

    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = threads
    sess_options.inter_op_num_threads = threads
    session = ort.InferenceSession(
        str(model_path),
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )
    _logger.info(f"Loaded ONNX model from {model_path} (threads={threads})")

    # ---------------------------------------------------------------------------
    # Data (RAW features: preprocessing is baked into the ONNX graph)
    #
    # Batch size 1: one event per inference, matching how MuonIdProducer would
    # invoke the model per event in production.
    # ---------------------------------------------------------------------------
    dataset = TrackerTrackSelectionDataset(
        path=config.paths[f"{split}_file"],
        config=config.data,
        max_events=max_events,
    )
    _logger.info(f"Number of events: {len(dataset)}")
    if len(dataset) <= warmup:
        raise ValueError(
            f"Need more than warmup={warmup} events, got {len(dataset)}"
        )

    loader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.data_load.num_workers,
        collate_fn=dataset.collate,
    )


    #---------------------------------------------------------------------------
    # Warmup loop: run the first `warmup` events to absorb cold-start latency and
    # allocator warmup. These events are not timed or recorded, but they do run
    # the same code paths as the timed loop to ensure a realistic warmup.
    #---------------------------------------------------------------------------
    for index, batch in enumerate(
        tqdm.rich.tqdm(loader, desc="Warmup", total=warmup)
    ):
        feeds = {name: batch[name].numpy() for name in INPUT_NAMES}
        session.run(OUTPUT_NAMES, feeds)
        if index < warmup:
            continue

    #---------------------------------------------------------------------------
    # Timing loop. Only session.run() is wrapped by the timer; feed-dict
    # construction is excluded.
    #---------------------------------------------------------------------------
    latency_ns_list: list[int] = []
    counts: dict[str, list[int]] = {name: [] for name in OBJECT_NAMES}

    for index, batch in enumerate(
        tqdm.rich.tqdm(loader, desc="Benchmarking", total=len(dataset))
    ):
        feeds = {name: batch[name].numpy() for name in INPUT_NAMES}

        start = time.perf_counter_ns()
        session.run(OUTPUT_NAMES, feeds)
        latency_ns = time.perf_counter_ns() - start

        latency_ns_list.append(latency_ns)
        for name in OBJECT_NAMES:
            counts[name].append(int(batch[f"{name}_data_mask"].sum()))

    latency_ns = np.asarray(latency_ns_list, dtype=np.int64)

    # ---------------------------------------------------------------------------
    # Per-event records -> HDF5
    # ---------------------------------------------------------------------------
    with h5.File(h5_path, "w") as h5_file:
        h5_file.create_dataset("latency_ns", data=latency_ns)
        for name in OBJECT_NAMES:
            h5_file.create_dataset(
                f"n_{name}", data=np.asarray(counts[name], dtype=np.int32)
            )
    _logger.info(f"Saved per-event latencies to {h5_path}")

    # ---------------------------------------------------------------------------
    # Metadata -> JSON sidecar
    # ---------------------------------------------------------------------------
    metadata = {
        "model_path": str(model_path),
        "split": split,
        "warmup": warmup,
        "max_events": max_events,
        "n_events_recorded": int(len(latency_ns)),
        "timestamp": datetime.now().isoformat(),
        "onnxruntime": {
            "version": ort.__version__,
            "device": ort.get_device(),
            "providers": ["CPUExecutionProvider"],
            "threads_cli": threads,
            "intra_op_num_threads": sess_options.intra_op_num_threads,
            "inter_op_num_threads": sess_options.inter_op_num_threads,
            "torch_num_threads": torch.get_num_threads(),
        },
        "cpu": collect_cpu_info(),
        "latency_summary": summarize_latency(latency_ns),
    }
    with open(json_path, "w") as json_file:
        json.dump(metadata, json_file, indent=2)
    _logger.info(f"Saved benchmark metadata to {json_path}")

    summary = metadata["latency_summary"]
    _logger.info(
        f"Latency over {summary['count']} events: "
        f"median={summary['median_ns'] / 1e3:.1f} us, "
        f"p95={summary['p95_ns'] / 1e3:.1f} us"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure ONNX inference latency at batch size 1.",
        formatter_class=ArgumentDefaultsRichHelpFormatter,
    )
    parser.add_argument(
        "-m",
        "--model",
        dest="model_path",
        type=Path,
        required=True,
        help="Path to the exported ONNX model (e.g., 'runs/exp1/model.onnx')",
    )
    parser.add_argument(
        "-s",
        "--split",
        dest="split",
        type=str,
        choices=["train", "val", "test"],
        default="test",
        help="Which data split to benchmark on (e.g., 'test')",
    )
    parser.add_argument(
        "-t",
        "--threads",
        dest="threads",
        type=int,
        default=4,
        help="Max CPU threads for ONNXRuntime op pools and torch",
    )
    parser.add_argument(
        "--warmup",
        dest="warmup",
        type=int,
        default=20,
        help="Number of leading events to discard (cold start / warmup)",
    )
    parser.add_argument(
        "--max-events",
        dest="max_events",
        type=int,
        default=None,
        help="Cap on number of events to load (default: all)",
    )
    args, unknown_args = parser.parse_known_args()

    run(**vars(args), args_list=unknown_args)


if __name__ == "__main__":
    main()
