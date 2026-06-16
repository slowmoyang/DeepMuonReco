import argparse
import logging
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


def extend_hdf5_dataset(dataset: h5.Dataset, new_data: np.ndarray) -> None:
    """Extend an HDF5 dataset with new data."""
    new_size = dataset.shape[0] + len(new_data)
    dataset.resize((new_size,))
    dataset[-len(new_data) :] = new_data


def run(
    model_path: Path,
    split: str,
    threads: int,
    args_list: list[str],
) -> None:
    """ """
    run_dir = model_path.parent
    output_dir = run_dir / "predictions"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"{split}-onnx.h5"

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
    # ---------------------------------------------------------------------------
    dataset = TrackerTrackSelectionDataset(
        path=config.paths[f"{split}_file"],
        config=config.data,
        max_events=config.data_load[f"{split}_max_events"],
    )
    _logger.info(f"Number of test examples: {len(dataset)}")

    loader = DataLoader(
        dataset=dataset,
        batch_size=config.data_load.eval_batch_size,
        shuffle=False,
        num_workers=config.data_load.num_workers,
        collate_fn=dataset.collate,
    )

    # --------------------------------------------------------------------------
    # output file
    # --------------------------------------------------------------------------
    output_file = h5.File(output_path, "w")
    output_file.create_dataset(
        name="score",
        shape=(0,),
        maxshape=(None,),
        dtype=h5.vlen_dtype(np.float32),
    )

    # --------------------------------------------------------------------------
    # Inference
    # --------------------------------------------------------------------------
    for batch in tqdm.rich.tqdm(loader, desc="Predicting"):
        feeds = {name: batch[name].numpy() for name in INPUT_NAMES}
        logits = session.run(OUTPUT_NAMES, feeds)[0]

        mask_batch = batch["tracker_track_data_mask"].bool().numpy()
        score_batch = torch.from_numpy(logits).sigmoid().float().numpy()

        score_batch = [score[mask] for mask, score in zip(mask_batch, score_batch)]
        score_batch = np.array(score_batch, dtype=object)

        extend_hdf5_dataset(output_file["score"], score_batch)

    output_file.close()
    _logger.info(f"Saved predictions to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Save ONNX model predictions to an HDF5 file.",
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
        help="Which data split to predict on (e.g., 'test')",
    )
    parser.add_argument(
        "-t",
        "--threads",
        dest="threads",
        type=int,
        default=4,
        help="Max CPU threads for ONNXRuntime op pools and torch",
    )
    args, unknown_args = parser.parse_known_args()

    run(**vars(args), args_list=unknown_args)


if __name__ == "__main__":
    main()
