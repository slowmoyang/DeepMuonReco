from typing import cast
import uproot
import numpy as np
import awkward as ak
import torch
from torch.utils.data import Dataset
from tensordict import TensorDict
from tensordict import pad_sequence


class InnerTrackSelectionDataset(Dataset):

    TRACK_FEATURE_LIST = [
        'px', 'py', 'eta',
    ]

    SEGMENT_FEATURE_LIST = [
        'position_x', 'position_y', 'position_z',
        'direction_x', 'direction_y', 'direction_z',
    ]

    DIM_TRACK = len(TRACK_FEATURE_LIST)
    DIM_SEGMENT = len(SEGMENT_FEATURE_LIST)
    DIM_TARGET = 1

    def __init__(self, path):
        self.example_list = self.process(path)

    def __getitem__(self, index: int) -> TensorDict:
        return self.example_list[index]

    def __len__(self) -> int:
        return len(self.example_list)

    @classmethod
    def process(
        cls,
        path,
        treepath: str = 'muons1stStep/event',
    ) -> list[TensorDict]:
        with uproot.open(path) as file: # type: ignore
            tree = cast(uproot.TTree, file[treepath])
            chunk = tree.arrays(library='ak')

        # NOTE: cut
        track_count = ak.count(chunk.track_pt, axis=1)
        dt_segment_count = ak.count(chunk.dt_segment_direction_x, axis=1)
        csc_segment_count = ak.count(chunk.csc_segment_direction_x, axis=1)
        segment_count = dt_segment_count + csc_segment_count
        chunk = chunk[(track_count > 0) & (segment_count > 0)]

        # tracker tracks
        chunk['track_px'] = chunk['track_pt'] * np.cos(chunk['track_phi'])
        chunk['track_py'] = chunk['track_pt'] * np.sin(chunk['track_phi'])

        track_chunk = [
            chunk[f'track_{feature}']
            for feature in cls.TRACK_FEATURE_LIST
        ]
        track_chunk = [
            torch.tensor(
                data=ak.to_numpy(np.stack(each, axis=1)),
                dtype=torch.float,
            )
            for each in zip(*track_chunk)
        ]

        # track segments in muon detectors
        segment_chunk = [
            np.concatenate([
                chunk[f'dt_segment_{feature}'],
                chunk[f'csc_segment_{feature}'],
            ])
            for feature in cls.SEGMENT_FEATURE_LIST
        ]

        segment_chunk = [
            torch.tensor(
                data=ak.to_numpy(np.stack(each, axis=1)),
                dtype=torch.float
            )
            for each in zip(*segment_chunk)
        ]

        target_chunk = [
            torch.tensor(ak.to_numpy(each), dtype=torch.float)
            for each in chunk['is_trackermuon']
        ]

        example_chunk: list[TensorDict] = []

        zipped = zip(track_chunk, segment_chunk, target_chunk)
        for track, segment, target in zipped:
            example = TensorDict(
                source=dict(
                    track=track,
                    segment=segment,
                    target=target,
                ),
                batch_size=[],
            )
            example_chunk.append(example)
        return example_chunk


    @classmethod
    def collate(cls, example_list: list[TensorDict]) -> TensorDict:
        batch = pad_sequence(example_list, return_mask=True)
        batch['pad_masks'] = TensorDict(
            source=dict(
                track=batch['masks']['track'].logical_not(),
                segment=batch['masks']['segment'].logical_not(),
            ),
            batch_size=batch.batch_size,
        )
        return batch
