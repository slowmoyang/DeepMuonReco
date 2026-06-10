# DeepMuonReco Ntuple and Dataset Production

## Overview

The training and evaluation datasets start from CMS GEN-SIM-RECO events. The
[`DeepMuonRecoNtuplizer`](https://github.com/eigen1907/DeepMuonRecoSample/blob/main/Ntuplizer/plugins/DeepMuonRecoNtuplizer.cc)
CMSSW analyzer reads reconstructed tracker tracks and muon-system objects,
constructs reconstruction and simulation labels for each track, and writes one
ROOT `TTree` entry per event. The resulting ROOT files are converted to HDF5
with [`scripts/convert-root-to-hdf5.py`](../scripts/convert-root-to-hdf5.py)
before they are used for deep-learning training and evaluation.

The data flow is:

```text
CMS GEN-SIM-RECO
  -> DeepMuonRecoNtuplizer in cmsRun
  -> ROOT file: deepMuonRecoNtuplizer/tree
  -> scripts/convert-root-to-hdf5.py
  -> HDF5 file with one variable-length dataset per ROOT branch
  -> TrackerTrackSelectionDataset
```

Each event contains variable numbers of tracker tracks, tracking particles,
muon-system hits, and muon-system segments. Branches belonging to the same
object collection are aligned by index within an event. For example,
`track_px[event][i]`, `track_charge[event][i]`, and
`track_is_trk_muon[event][i]` all describe the same tracker track.

## Producing the ROOT Ntuple

The analyzer consumes the following collections by default:

| Configuration parameter | Default input tag | Purpose |
| --- | --- | --- |
| `muons` | `muons` | Reconstructed muons used to label tracker tracks |
| `tracks` | `generalTracks` | Reconstructed tracker tracks and classification examples |
| `trackingParticles` | `mix:MergedTrackTruth` | Simulated tracking particles |
| `associator` | `quickTrackAssociatorByHits` | Reconstructed-track to tracking-particle association |
| `rpcRecHits` | `rpcRecHits` | RPC reconstructed hits |
| `gemRecHits` | `gemRecHits` | GEM reconstructed hits |
| `gemSegments` | `gemSegments` | GEM/ME0 reconstructed segments |
| `dtSegments` | `dt4DSegments` | DT four-dimensional reconstructed segments |
| `cscSegments` | `cscSegments` | CSC reconstructed segments |

The upstream example configuration,
[`runDeepMuonRecoNtuplizer_cfg.py`](https://github.com/eigen1907/DeepMuonRecoSample/blob/main/Ntuplizer/test/runDeepMuonRecoNtuplizer_cfg.py),
loads the Phase-2 geometry and magnetic field, configures the GlobalTag, runs
the tracking-particle cluster producer and hit-based track associator, and then
runs the ntuplizer. A typical invocation is:

```bash
cmsRun DeepMuonRecoSample/Ntuplizer/test/runDeepMuonRecoNtuplizer_cfg.py \
  inputFiles=file:/path/to/input_GEN-SIM-RECO.root \
  outputFile=deep-muon-reco.root \
  maxEvents=-1
```

The CMSSW release, detector geometry, GlobalTag, and collection input tags must
match the GEN-SIM-RECO campaign being processed. The upstream example currently
uses `GeometryExtended2026D110Reco_cff` and
`140X_mcRun4_realistic_v4`; these are examples rather than universal settings.

`TFileService` places the tree at `deepMuonRecoNtuplizer/tree`. At the start of
each event, the analyzer clears all output vectors. It then fills reconstructed
muon-system objects, tracker tracks and their labels, and tracking particles
before calling `TTree::Fill()`.

## Track Labels

Every track from the configured `tracks` collection is saved. No track is
removed by the ntuplizer. Instead, selection and target information are stored
as per-track integer branches.

### Good-track selection

`track_is_good_track` is `1` when the track satisfies all of:

```text
track pt > 0.5 GeV
track p  > 2.5 GeV
abs(track eta) < 3.0
```

Otherwise it is `0`. This reproduces the kinematic part of the initial tracker
track selection used for tracker-muon reconstruction. The current dataset
loader uses this branch as a mask before training.

### Reconstructed-muon labels

The analyzer builds a map from each reconstructed muon's `innerTrack().key()`
to that muon. A tracker track is labeled as a reconstructed muon when its key
appears in this map:

- `track_is_reco_muon`: the track is the inner track of a reconstructed muon.
- `track_is_trk_muon`: that reconstructed muon has `isTrackerMuon()` set.
- `track_is_glb_muon`: that reconstructed muon has `isGlobalMuon()` set.
- `track_is_pf_muon`: that reconstructed muon has `isPFMuon()` set.

This key-based labeling assumes that the muon inner tracks refer to the
configured `tracks` collection. The input tags must be chosen consistently;
otherwise equal numeric keys from different track products could produce
incorrect labels.

The current binary-classification target is `track_is_trk_muon`. It represents
the tracker-muon type bit present in the input reconstructed-muon collection.
See [MuonIdProducer: Tracker Muon Identification Logic](muon-id-producer.md)
for how that bit is assigned and may be removed after arbitration.

### Simulation matching labels

`quickTrackAssociatorByHits` associates reconstructed tracks to tracking
particles. When associations exist, the ntuplizer takes the first association
returned for the track and stores its quality and tracking-particle index.
`track_is_matched_muon` is `1` only when that tracking particle has
`abs(pdgId) == 13` and `status == 1`.

An unmatched track has:

```text
track_is_matched_muon = 0
track_match_tp_idx = -1
track_match_quality = 0
```

## Coordinate and Collection Conventions

- Tracker-track kinematics and reference-point coordinates come directly from
  `reco::Track`.
- Hit and segment positions are transformed from detector-local coordinates to
  CMS global Cartesian coordinates using the corresponding detector geometry.
- Segment directions are also transformed to global Cartesian coordinates.
- Stored position and direction uncertainties are square roots of diagonal
  local-error components. They are not transformed to global-coordinate
  uncertainties.
- Position values are conventionally in centimeters. Direction components are
  dimensionless.
- `rawid` is the packed CMS detector geographical identifier.
- GEM reconstructed hits include GE1/1 and GE2/1 but explicitly exclude ME0.
- `gem_seg_*` branches contain only ME0 segments; non-ME0 GEM segments are
  explicitly excluded.
- Missing or invalid hit/segment collections produce empty vectors for that
  collection in the event.

## Branch Reference

All branches are vectors. In ROOT, most numeric feature vectors use
`std::vector<double>`, labels and identifiers use `std::vector<int>`, and
`track_match_quality` uses `std::vector<float>`.

### Tracker tracks

| Branch | ROOT element type | Description |
| --- | --- | --- |
| `track_pt` | `double` | Transverse momentum |
| `track_eta` | `double` | Pseudorapidity |
| `track_phi` | `double` | Azimuthal angle |
| `track_px`, `track_py`, `track_pz` | `double` | Global Cartesian momentum components |
| `track_vx`, `track_vy`, `track_vz` | `double` | Track reference-point coordinates |
| `track_qoverp` | `double` | Signed charge divided by momentum magnitude |
| `track_lamda` | `double` | Track dip angle from `reco::Track::lambda()` |
| `track_dxy` | `double` | Transverse impact parameter at the reference point |
| `track_dsz` | `double` | Longitudinal impact-parameter quantity at the reference point |
| `track_qoverp_err` | `double` | Uncertainty on `qoverp` |
| `track_lambda_err` | `double` | Uncertainty on `lambda` |
| `track_phi_err` | `double` | Uncertainty on `phi` |
| `track_dxy_err` | `double` | Uncertainty on `dxy` |
| `track_dsz_err` | `double` | Uncertainty on `dsz` |
| `track_charge` | `int` | Electric charge |
| `track_chi2` | `double` | Track-fit chi-squared |
| `track_ndof` | `double` | Track-fit number of degrees of freedom |
| `track_n_algo` | `int` | Numeric `reco::TrackBase::TrackAlgorithm` value |
| `track_is_good_track` | `int` | Good-track kinematic selection flag |
| `track_is_reco_muon` | `int` | Track is an inner track of a reconstructed muon |
| `track_is_trk_muon` | `int` | Associated reconstructed muon is a tracker muon |
| `track_is_glb_muon` | `int` | Associated reconstructed muon is a global muon |
| `track_is_pf_muon` | `int` | Associated reconstructed muon is a particle-flow muon |
| `track_is_matched_muon` | `int` | Best associated tracking particle is a stable simulated muon |
| `track_match_quality` | `float` | Quality of the selected reconstructed-track to tracking-particle association |
| `track_match_tp_idx` | `int` | Index into the event's `tp_*` arrays, or `-1` when unmatched |

The branch name `track_lamda` is misspelled in the ROOT schema. The related
uncertainty branch is spelled `track_lambda_err`.

### Tracking particles

| Branch | ROOT element type | Description |
| --- | --- | --- |
| `tp_pt` | `double` | Tracking-particle transverse momentum |
| `tp_eta` | `double` | Tracking-particle pseudorapidity |
| `tp_phi` | `double` | Tracking-particle azimuthal angle |
| `tp_pdg_id` | `int` | PDG particle identifier |
| `tp_charge` | `int` | Electric charge |
| `tp_status` | `int` | Generator status stored by the tracking particle |

### RPC and GEM reconstructed hits

The following suffixes exist for both the `rpc_hit_` and `gem_hit_` prefixes:

| Branch suffix | ROOT element type | Description |
| --- | --- | --- |
| `rawid` | `int` | Packed detector geographical identifier |
| `pos_x`, `pos_y`, `pos_z` | `double` | Global hit-position components |
| `pos_x_err`, `pos_y_err` | `double` | Square roots of local position-error `xx` and `yy` components |
| `cls` | `int` | Hit cluster size |
| `bx` | `int` | Hit bunch crossing |

Thus, for example, `rpc_hit_pos_x` is the global x coordinate of each RPC hit,
while `gem_hit_bx` is the bunch crossing of each non-ME0 GEM hit.

### GEM, DT, and CSC reconstructed segments

The following suffixes exist for the `gem_seg_`, `dt_seg_`, and `csc_seg_`
prefixes:

| Branch suffix | ROOT element type | Description |
| --- | --- | --- |
| `rawid` | `int` | Packed detector geographical identifier |
| `pos_x`, `pos_y`, `pos_z` | `double` | Global segment-position components |
| `pos_x_err`, `pos_y_err` | `double` | Square roots of local position-error `xx` and `yy` components |
| `dir_x`, `dir_y`, `dir_z` | `double` | Global segment-direction components |
| `dir_x_err`, `dir_y_err` | `double` | Square roots of local direction-error `xx` and `yy` components |
| `chi2` | `double` | Segment-fit chi-squared |
| `ndof` | `double` | Segment-fit number of degrees of freedom |

CSC segments additionally have:

| Branch | ROOT element type | Description |
| --- | --- | --- |
| `csc_seg_time` | `double` | Reconstructed CSC segment time |

## Converting ROOT to HDF5

Convert one or more ROOT files with:

```bash
python scripts/convert-root-to-hdf5.py \
  --input /path/to/train.root /path/to/validation.root
```

The converter defaults to the tree path `deepMuonRecoNtuplizer/tree` and reads
the ROOT tree in `4 GB` batches. Both settings can be overridden:

```bash
python scripts/convert-root-to-hdf5.py \
  --input /path/to/input.root \
  --treepath deepMuonRecoNtuplizer/tree \
  --step-size "1 GB"
```

For each input ROOT file, the script creates a sibling file with the `.h5`
suffix. For example, `train.root` becomes `train.h5`. It refuses to overwrite
an existing output file. Supplying multiple inputs converts each file
independently; the script does not merge files or create train, validation, and
test splits.

Every ROOT branch becomes a top-level HDF5 dataset with the same name. The
first HDF5 dimension is the event index, and each event element is a
variable-length array of objects. Variable-length arrays whose ROOT data are
read as `float64` are converted to `float32`; integer arrays and
`track_match_quality` retain their input types. Batches are appended to
resizable, chunked HDF5 datasets.

The converter expects each batch to contain at least one event and uses the
first event's array to infer each variable-length dataset's element type. It
does not filter tracks, remove empty events, or perform other event cleaning.
Any cleaning or dataset splitting must therefore happen in a separate step.

## Branches Used for Training

The ntuple contains more information than the current models consume.
[`config/data/mu2030pu.yaml`](../config/data/mu2030pu.yaml) selects:

| Model object | Ntuple branches used as features |
| --- | --- |
| Tracker track | `track_px`, `track_py`, `track_pz`, `track_vx`, `track_vy`, `track_vz`, `track_charge` |
| DT segment | `dt_seg_pos_x`, `dt_seg_pos_y`, `dt_seg_pos_z`, `dt_seg_dir_x`, `dt_seg_dir_y`, `dt_seg_dir_z` |
| CSC segment | `csc_seg_pos_x`, `csc_seg_pos_y`, `csc_seg_pos_z`, `csc_seg_dir_x`, `csc_seg_dir_y`, `csc_seg_dir_z` |
| GEM/ME0 segment | `gem_seg_pos_x`, `gem_seg_pos_y`, `gem_seg_pos_z`, `gem_seg_dir_x`, `gem_seg_dir_y`, `gem_seg_dir_z` |
| RPC hit | `rpc_hit_pos_x`, `rpc_hit_pos_y`, `rpc_hit_pos_z` |
| GEM hit | `gem_hit_pos_x`, `gem_hit_pos_y`, `gem_hit_pos_z` |

The loader applies `track_is_good_track` as a tracker-track mask and uses
`track_is_trk_muon` as the target. The same mask is applied to tracker-track
features, `track_pt`, and the target so their per-event indices remain aligned.
Muon-system objects are not filtered by the loader. During batching, all
variable-length collections are padded and accompanied by data masks.
