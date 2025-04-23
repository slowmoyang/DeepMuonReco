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

    RECHIT_FEATURE_LIST = [
        'position_x', 'position_y', 'position_z',
    ]

    DIM_TRACK = len(TRACK_FEATURE_LIST)
    DIM_SEGMENT = len(SEGMENT_FEATURE_LIST)
    DIM_RECHIT = len(RECHIT_FEATURE_LIST)
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
        path: str,
        treepath: str = 'muons1stStep/event',
    ) -> list[TensorDict]:
        with uproot.open(path) as file: # type: ignore
            tree = cast(uproot.TTree, file[treepath])
            chunk = tree.arrays(library='ak')

        track_count = ak.count(chunk.track_pt, axis=1)
        dt_segment_count = ak.count(chunk.dt_segment_direction_x, axis=1)
        csc_segment_count = ak.count(chunk.csc_segment_direction_x, axis=1)
        segment_count = dt_segment_count + csc_segment_count

        rpc_rechit_count = ak.count(chunk.rpc_rechit_position_x, axis=1)
        gem_rechit_count = ak.count(chunk.gem_rechit_position_x, axis=1)
        rechit_count = rpc_rechit_count + gem_rechit_count

        muon_obj_count = segment_count + rechit_count
        chunk = chunk[(track_count > 0) & (muon_obj_count > 0)]

        nevents = len(chunk.track_pt)
        track_list, segment_list, rechit_list, target_list = [], [], [], []
        for i in range(nevents):
            # --- tracks: [pt, eta, phi] to [px, py, eta]
            pt  = chunk.track_pt[i]
            phi = chunk.track_phi[i]
            eta = chunk.track_eta[i]
            px = pt * np.cos(phi)
            py = pt * np.sin(phi)
            track_arr = np.stack([px, py, eta], axis=1)  # (n_trk, 3)
            track_list.append(torch.tensor(track_arr, dtype=torch.float))

            # --- segment (DT + CSC): stack [pos_x, pos_y, pos_z, dir_x, dir_y, dir_z]
            def build_segment(prefix: str):
                return np.stack([
                    getattr(chunk, f'{prefix}_position_x')[i],
                    getattr(chunk, f'{prefix}_position_y')[i],
                    getattr(chunk, f'{prefix}_position_z')[i],
                    getattr(chunk, f'{prefix}_direction_x')[i],
                    getattr(chunk, f'{prefix}_direction_y')[i],
                    getattr(chunk, f'{prefix}_direction_z')[i],
                ], axis=1) if len(getattr(chunk, f'{prefix}_position_x')[i]) > 0 else np.zeros((0, 6))
            dt_arr  = build_segment('dt_segment')
            csc_arr = build_segment('csc_segment')
            segment_arr = np.concatenate([dt_arr, csc_arr], axis=0)  # (n_dt+n_csc,6)
            segment_list.append(torch.tensor(segment_arr, dtype=torch.float))

            # --- rechit (RPC + GEM): stack ([pos_x, pos_y, pos_z])
            def build_rec(prefix: str):
                return np.stack([
                    getattr(chunk, f'{prefix}_position_x')[i],
                    getattr(chunk, f'{prefix}_position_y')[i],
                    getattr(chunk, f'{prefix}_position_z')[i],
                ], axis=1) if len(getattr(chunk, f'{prefix}_position_x')[i]) > 0 else np.zeros((0, 3))
            rpc_arr = build_rec('rpc_rechit')
            gem_arr = build_rec('gem_rechit')
            rec_arr = np.concatenate([rpc_arr, gem_arr], axis=0)  # (n_rpc+n_gem,3)
            rechit_list.append(torch.tensor(rec_arr, dtype=torch.float))

            # --- target (is_trackermuon): [is_trackermuon] -> [is_trackermuon]
            tgt = ak.to_numpy(chunk.is_trackermuon[i])
            target_list.append(torch.tensor(tgt, dtype=torch.float))

        example_chunk: list[TensorDict] = []
        for track, seg, rec, tgt in zip(track_list, segment_list, rechit_list, target_list):
            example_chunk.append(
                TensorDict(
                    source=dict(
                        track=track,
                        segment=seg,
                        rechit=rec,
                        target=tgt,
                    ),
                    batch_size=[],
                )
            )
        return example_chunk

    @classmethod
    def collate(cls, example_list: list[TensorDict]) -> TensorDict:
        batch = pad_sequence(example_list, return_mask=True)
        batch['pad_masks'] = TensorDict(
            source=dict(
                track=batch['masks']['track'].logical_not(),
                segment=batch['masks']['segment'].logical_not(),
                rechit=batch['masks']['rechit'].logical_not(),
            ),
            batch_size=batch.batch_size,
        )
        return batch
