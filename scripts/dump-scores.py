"""Dump a flat per-track score table for offline differential analysis.

One GPU pass over a split produces a single table joining the model score with
the track kinematics and the truth branches that are not part of the training
tensors (``track_is_matched_muon``, ``track_is_reco_muon``, ...). Every
subsequent diagnostic — per-pT-bin ROC, background composition, ghost-track
matching — then runs offline from this file.

Rows are ordered event by event, and within an event by track index after the
``track_is_good_track`` selection, which is the same order the dataset and the
model use. The script verifies that alignment by cross-checking the pT read
back from HDF5 against the pT carried through the model batch.
"""

import argparse
import logging
from pathlib import Path
from contextlib import nullcontext
import warnings

import h5py as h5
import numpy as np
import torch
from torch.utils.data import DataLoader

from tensordict.nn import TensorDictModule

from hydra.utils import instantiate
from omegaconf import OmegaConf
from omegaconf import DictConfig

import tqdm.rich
from tqdm import TqdmExperimentalWarning
from rich.logging import RichHandler
from rich_argparse import ArgumentDefaultsRichHelpFormatter

from muonly.data.datasets import TrackerTrackSelectionDataset
from muonly.data.utils import configure_model_in_keys
from muonly.data.transforms.utils import configure_preprocessing

_logger = logging.getLogger(Path(__file__).name)
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()],
)

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)


# Branches read straight from the source HDF5 and joined onto the model output.
# ``track_pt`` is included as the alignment check against the model batch.
TRUTH_BRANCHES: dict[str, str] = {
    "pt_check": "track_pt",
    "eta": "track_eta",
    "phi": "track_phi",
    "is_matched_muon": "track_is_matched_muon",
    "is_reco_muon": "track_is_reco_muon",
    "is_glb_muon": "track_is_glb_muon",
    "is_pf_muon": "track_is_pf_muon",
    "chi2": "track_chi2",
    "ndof": "track_ndof",
}

COLUMN_DTYPES: dict[str, np.dtype] = {
    "score": np.dtype(np.float32),
    "pt": np.dtype(np.float32),
    "target": np.dtype(np.int8),
    "event_index": np.dtype(np.int32),
    "eta": np.dtype(np.float32),
    "phi": np.dtype(np.float32),
    "is_matched_muon": np.dtype(np.int8),
    "is_reco_muon": np.dtype(np.int8),
    "is_glb_muon": np.dtype(np.int8),
    "is_pf_muon": np.dtype(np.int8),
    "chi2": np.dtype(np.float32),
    "ndof": np.dtype(np.float32),
}


def read_good_track_branches(
    path: Path,
    n_events: int,
    apply_good_track_mask: bool,
) -> dict[str, np.ndarray]:
    """Read the extra truth branches under the same good-track selection.

    Mirrors ``TrackerTrackSelectionDataset.from_hdf5`` so the flattened rows
    line up with the model output.
    """
    slicing = slice(None, n_events)
    columns: dict[str, list[np.ndarray]] = {name: [] for name in TRUTH_BRANCHES}
    event_index: list[np.ndarray] = []

    with h5.File(path, "r") as file:
        if apply_good_track_mask:
            masks = [each.astype(bool) for each in file["track_is_good_track"][slicing]]
        else:
            masks = None

        raw = {
            name: file[branch][slicing] for name, branch in TRUTH_BRANCHES.items()
        }

    for index in range(n_events):
        mask = masks[index] if masks is not None else None
        for name in TRUTH_BRANCHES:
            values = raw[name][index]
            columns[name].append(values[mask] if mask is not None else values)
        n_tracks = len(columns["pt_check"][-1])
        event_index.append(np.full(n_tracks, index, dtype=np.int32))

    flat = {name: np.concatenate(values) for name, values in columns.items()}
    flat["event_index"] = np.concatenate(event_index)
    return flat


@torch.inference_mode()
def run(
    ckpt_path: Path,
    split: str,
    output_path: Path | None,
    args_list: list[str],
) -> None:
    run_dir = ckpt_path.parents[1]

    base_config = OmegaConf.load(run_dir / "config.yaml")
    cli_config = OmegaConf.from_cli(args_list)
    config = OmegaConf.merge(base_config, cli_config)
    if not isinstance(config, DictConfig):
        raise ValueError("Expected config to be a DictConfig")

    if output_path is None:
        output_dir = run_dir / "scores"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{split}.h5"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(config.torch.device)
    _logger.info(f"{device=}")

    # ---------------------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------------------
    model = instantiate(config.model)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    del checkpoint

    model = model.to(device).eval()
    _logger.info(f"Loaded checkpoint from {ckpt_path}")

    td_model = TensorDictModule(
        module=model,
        in_keys=configure_model_in_keys(config=config),
        out_keys=["logits"],
    )

    # ---------------------------------------------------------------------------
    # Data
    # ---------------------------------------------------------------------------
    source_path = Path(config.paths[f"{split}_file"])
    dataset = TrackerTrackSelectionDataset(
        path=source_path,
        config=config.data,
        max_events=config.data_load[f"{split}_max_events"],
    )
    n_events = len(dataset)
    _logger.info(f"Number of {split} events: {n_events}")

    preprocessing = configure_preprocessing(config.data)
    dataset.apply_(preprocessing)

    loader = DataLoader(
        dataset=dataset,
        batch_size=config.data_load.eval_batch_size,
        shuffle=False,
        num_workers=config.data_load.num_workers,
        collate_fn=dataset.collate,
    )

    if config.torch.precision == "bfloat16":
        amp_context = torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    else:
        amp_context = nullcontext()

    # ---------------------------------------------------------------------------
    # Inference
    # ---------------------------------------------------------------------------
    score_chunks: list[np.ndarray] = []
    pt_chunks: list[np.ndarray] = []
    target_chunks: list[np.ndarray] = []

    for batch in tqdm.rich.tqdm(loader, desc="Scoring"):
        batch = batch.to(device)
        with amp_context:
            batch = td_model(batch)

        mask = batch["tracker_track_data_mask"]
        score_chunks.append(batch["logits"][mask].sigmoid().float().cpu().numpy())
        pt_chunks.append(batch["tracker_track_pt"][mask].float().cpu().numpy())
        target_chunks.append(batch["target"][mask].to(torch.int8).cpu().numpy())

    table: dict[str, np.ndarray] = {
        "score": np.concatenate(score_chunks),
        "pt": np.concatenate(pt_chunks),
        "target": np.concatenate(target_chunks),
    }
    del score_chunks, pt_chunks, target_chunks

    n_tracks = table["score"].size
    _logger.info(f"Scored {n_tracks:,} tracks over {n_events:,} events.")

    # ---------------------------------------------------------------------------
    # Join the truth branches and verify the row alignment
    # ---------------------------------------------------------------------------
    _logger.info(f"Reading truth branches from {source_path} ...")
    truth = read_good_track_branches(
        path=source_path,
        n_events=n_events,
        apply_good_track_mask=bool(config.data.tracker_track.is_good),
    )

    if truth["event_index"].size != n_tracks:
        raise RuntimeError(
            f"Row count mismatch: {n_tracks:,} scored tracks vs "
            f"{truth['event_index'].size:,} tracks read from HDF5. The "
            f"good-track selection or the event order does not match."
        )

    pt_deviation = np.abs(truth.pop("pt_check").astype(np.float32) - table["pt"]).max()
    if pt_deviation > 1e-3:
        raise RuntimeError(
            f"Row alignment check failed: max |pt_hdf5 - pt_batch| = "
            f"{pt_deviation:.6g}. Rows are not in the same order."
        )
    _logger.info(f"Row alignment verified (max pT deviation {pt_deviation:.3g}).")

    table.update(truth)

    # ---------------------------------------------------------------------------
    # Write
    # ---------------------------------------------------------------------------
    with h5.File(output_path, "w") as file:
        for name, values in table.items():
            file.create_dataset(
                name=name,
                data=values.astype(COLUMN_DTYPES[name]),
                compression="lzf",
                chunks=True,
            )
        file.attrs["checkpoint"] = str(ckpt_path)
        file.attrs["split"] = split
        file.attrs["source"] = str(source_path)
        file.attrs["n_events"] = n_events

    _logger.info(f"Wrote {n_tracks:,} rows to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=ArgumentDefaultsRichHelpFormatter,
    )
    parser.add_argument(
        "-c",
        "--ckpt",
        dest="ckpt_path",
        type=Path,
        required=True,
        help="Path to the model checkpoint (e.g. 'logs/<exp>/<run>/checkpoints/best.pt')",
    )
    parser.add_argument(
        "-s",
        "--split",
        dest="split",
        type=str,
        choices=["train", "val", "test"],
        default="val",
        help="Which data split to score",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_path",
        type=Path,
        default=None,
        help="Output HDF5 path (default: <run_dir>/scores/<split>.h5)",
    )
    args, unknown_args = parser.parse_known_args()

    run(**vars(args), args_list=unknown_args)


if __name__ == "__main__":
    main()
