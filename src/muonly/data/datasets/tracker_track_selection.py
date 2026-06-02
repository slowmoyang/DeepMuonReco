from pathlib import Path
import logging
from typing import Self, cast
import json
import h5py as h5
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from tensordict import TensorDict
from omegaconf import DictConfig
from omegaconf import OmegaConf
import tqdm.rich
from ..transforms import Compose


__all__ = [
    "TrackerTrackSelectionDataset",
]


_logger = logging.getLogger(__name__)



def _stack_features(feature_list: list[np.ndarray]) -> list[torch.Tensor]:
    return [
        torch.from_numpy(np.stack(each, axis=1, dtype=np.float32))
        for each in zip(*feature_list)
    ]


class TrackerTrackSelectionDataset(Dataset):
    OBJECT_TYPES = (
        "tracker_track",
        "dt_segment",
        "csc_segment",
        "gem_segment",
        "rpc_hit",
        "gem_hit",
    )



    def __init__(
        self,
        path: str | Path,
        config: dict | DictConfig,
        max_events: int | float | None = None,
    ) -> None:
        super().__init__()

        if isinstance(config, DictConfig):
            config = OmegaConf.to_container(cfg=config, resolve=True)  # type: ignore
        if not isinstance(config, dict):
            raise TypeError("config must be a dict or DictConfig.")

        _logger.debug(f'Initializing {self.__class__.__name__} with {path=}, {max_events=}, and {config=}...')


        path = Path(path)

        _logger.debug(f"{path.suffix} file format detected.")
        if path.suffix == ".root":
            loader = self.from_root
        elif path.suffix in [".h5", ".hdf5"]:
            loader = self.from_hdf5
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        _logger.debug(f"Using {loader.__name__} to load the data.")

        def get_features(config, key):
            if (key not in config) or (config[key] is None):
                return None
            return config[key].get("features", None)

        tracker_track_features = get_features(config, "tracker_track")
        dt_segment_features = get_features(config, "dt_segment")
        csc_segment_features = get_features(config, "csc_segment")
        gem_segment_features = get_features(config, "gem_segment")
        rpc_hit_features = get_features(config, "rpc_hit")
        gem_hit_features = get_features(config, "gem_hit")

        if tracker_track_features is None:
            raise ValueError("tracker_track_features must be specified in the config.")
        if dt_segment_features is None:
            raise ValueError("dt_segment_features must be specified in the config.")
        if csc_segment_features is None:
            raise ValueError("csc_segment_features must be specified in the config.")

        _logger.debug(f"Loading examples from {path} ...")
        self.example_list = loader(
            path=path,
            tracker_track_features=tracker_track_features,
            dt_segment_features=dt_segment_features,
            csc_segment_features=csc_segment_features,
            gem_segment_features=gem_segment_features,
            rpc_hit_features=rpc_hit_features,
            gem_hit_features=gem_hit_features,
            target_key=config["tracker_track"]["target"],
            max_events=max_events,
            tracker_track_is_good=config['tracker_track']['is_good'],
        )
        _logger.info(f"Loaded {len(self.example_list)} examples from {path}.")

    def __getitem__(self, index: int) -> TensorDict:
        return self.example_list[index]

    def __len__(self) -> int:
        return len(self.example_list)

    @classmethod
    def _get_stop(cls, max_events: int | float | None, total: int) -> int | None:
        if max_events is None:
            stop = None
        elif isinstance(max_events, int):
            if max_events < 0:
                raise ValueError("max_events must be a non-negative integer or None.")
            stop = min(max_events, total)
        elif isinstance(max_events, float):
            if not (0.0 < max_events <= 1.0):
                raise ValueError("max_events must be a float in the range (0, 1].")
            stop = int(total * max_events)
            if stop < 1:
                raise ValueError("max_events results in an empty dataset.")
        else:
            raise TypeError("max_events must be an int, float, or None.")

        return stop

    @classmethod
    def from_root(
        cls,
        path: str | Path,
        tracker_track_features: list[str],
        dt_segment_features: list[str],
        csc_segment_features: list[str],
        gem_segment_features: list[str] | None,
        rpc_hit_features: list[str] | None,
        gem_hit_features: list[str] | None,
        max_events: int | float | None,
        tracker_track_is_good: bool,
        target_key: str = "track_is_trk_muon",
        treepath: str = "deepMuonRecoNtuplizer/tree",
    ) -> list[TensorDict]:
        raise NotImplementedError("Root file loading is not implemented yet.")

    @classmethod
    def from_hdf5(
        cls,
        path: str | Path,
        max_events: int | float | None,
        tracker_track_features: list[str],
        dt_segment_features: list[str],
        csc_segment_features: list[str],
        gem_segment_features: list[str] | None,
        rpc_hit_features: list[str] | None,
        gem_hit_features: list[str] | None,
        tracker_track_is_good: bool,
        target_key: str = "track_is_trk_muon",
    ) -> list[TensorDict]:
        """
        For the HDF5 files, we assume that event cleaning has already been
        performed: events without tracker tracks or without segments/hits in
        the muon detector have been removed. We also assume that px and py of
        each track are precomputed and stored.
        """

        with h5.File(path, "r") as file:
            total = len(file[next(iter(file.keys()))])  # type: ignore
            stop = cls._get_stop(max_events=max_events, total=total)
            slicing = slice(None, stop)

            chunk = {}

            # NOTE: reconstructed tracker tracks
            if tracker_track_is_good:
                _logger.debug("Using 'track_is_good_track' as a mask to select good tracker tracks.")
                mask = file['track_is_good_track'][slicing]
                mask = np.vectorize(lambda each: each.astype(np.bool_), otypes=[object])(mask)

                eff = np.mean(np.concatenate(mask))
                _logger.info(f"Using 'track_is_good_track' mask results in {eff:.2%} of the original tracker tracks being selected.")
            else:
                mask = None

            def select_tracker_tracks(arr: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
                if mask is None:
                    return arr
                return np.array(object=[x[m] for x, m in zip(arr, mask)], dtype=object)


            chunk["tracker_track"] = [
                select_tracker_tracks(file[f"track_{each}"][slicing], mask)  # type: ignore
                for each in tracker_track_features
            ]

            chunk['tracker_track_pt'] = [
                torch.from_numpy(each)
                for each in  select_tracker_tracks(file['track_pt'][slicing], mask)  # type: ignore
            ]

            chunk["target"] = [
                torch.from_numpy(each)
                for each in select_tracker_tracks(file[target_key][slicing], mask) # type: ignore
            ]

            # NOTE: reconstructed segments in the muon system
            chunk["dt_segment"] = [
                file[f"dt_seg_{each}"][slicing]  # type: ignore
                for each in dt_segment_features
            ]

            chunk["csc_segment"] = [
                file[f"csc_seg_{each}"][slicing]  # type: ignore
                for each in csc_segment_features
            ]

            if gem_segment_features is not None:
                chunk["gem_segment"] = [
                    file[f"gem_seg_{each}"][slicing]  # type: ignore
                    for each in gem_segment_features
                ]

            # NOTE: reconstructed hits in the muon system
            if rpc_hit_features is not None:
                chunk["rpc_hit"] = [
                    file[f"rpc_hit_{each}"][slicing]  # type: ignore
                    for each in rpc_hit_features
                ]

            if gem_hit_features is not None:
                chunk["gem_hit"] = [
                    file[f"gem_hit_{each}"][slicing]  # type: ignore
                    for each in gem_hit_features
                ]

        key_list_for_stack = ["tracker_track", "dt_segment", "csc_segment"]
        if gem_segment_features is not None:
            key_list_for_stack.append("gem_segment")
        if rpc_hit_features is not None:
            key_list_for_stack.append("rpc_hit")
        if gem_hit_features is not None:
            key_list_for_stack.append("gem_hit")

        for key in key_list_for_stack:
            chunk[key] = _stack_features(chunk[key])

        return [
            TensorDict(dict(zip(chunk.keys(), each))) for each in zip(*chunk.values())
        ]

    def summarize(
        self,
        path: Path | None = None,
        verbose: bool = False,
    ) -> dict[str, dict[str, float]]:
        """Log min/max/mean/std of per-event object counts for each object type."""
        if len(self) == 0:
            raise RuntimeError("Cannot summarize an empty dataset.")

        available = [
            key for key in self.OBJECT_TYPES if key in self.example_list[0].sorted_keys
        ]

        counts = {key: [] for key in available}
        for example in self.example_list:
            for key in available:
                counts[key].append(example[key].shape[0])

        log = {}
        for key, values in counts.items():
            tensor = torch.tensor(values, dtype=torch.float32)

            log[key] = {
                "min": tensor.min().item(),
                "max": tensor.max().item(),
                "mean": tensor.mean().item(),
                "std": tensor.std().item(),
            }

        if path is not None:
            with path.open(mode='w') as stream:
                json.dump(log, stream, indent=4)

        if verbose:
            for key, stats in log.items():
                _logger.info(
                    f"{key}: count min={stats['min']}, max={stats['max']}, mean={stats['mean']:.2f}, std={stats['std']:.2f}"
                )

        return log

    def apply_(self, transforms: dict[str, Compose]) -> Self:
        """Apply transforms transforms to all examples in-place.

        Args:
            transforms (dict): A dictionary where keys are object name (e.g., 'tracker_track', 'dt_segment', etc.) and values are lists of transforms to apply to that object.
        """
        # drop if the key is not found in the examples, and log a warning
        unavailable_keys = [
            key
            for key in transforms.keys()
            if key not in self.example_list[0].sorted_keys
        ]
        if len(unavailable_keys) > 0:
            for key in unavailable_keys:
                _logger.warning(
                    f"Key '{key}' not found in the examples. The corresponding transforms will be ignored."
                )
                del transforms[key]

        for key, value in transforms.items():
            _logger.debug(f"Applying the following transforms to {key}: {value}")

        _logger.debug(f"Applying transforms to {len(self.example_list)} examples ...")
        for example in tqdm.rich.tqdm(self.example_list):
            for key, value in transforms.items():
                example[key] = value(example[key])
        _logger.debug(f"Applied transforms to {len(self.example_list)} examples.")

        return self

    @classmethod
    def collate(cls, example_list: list[TensorDict]) -> TensorDict:
        batch_dict = {}
        for key in example_list[0].sorted_keys:
            tensors = [example[key] for example in example_list]
            padded = pad_sequence(tensors, batch_first=True, padding_value=0)
            batch_dict[key] = padded
            # we can use tracker_track_data_mask for target (tracker_track_target)
            if key in ["target", "tracker_track_pt"]:
                continue
            lengths = torch.tensor([t.shape[0] for t in tensors])
            max_len = padded.shape[1]
            mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)
            batch_dict[f"{key}_data_mask"] = mask
        return TensorDict(batch_dict)
