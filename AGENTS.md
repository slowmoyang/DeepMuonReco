# Repository Guidelines

## Documentations
Read the following documentation files for detailed information about the project if needed:

- `docs/project.md`: provide an overview of the project
- `docs/muon-id-producer.md` provides details about `MuonIdProducer`, which performs tracker muon reconstruction and provides labels for training and evaluation.
- `docs/ntuplizer.md` describes DeepMuonReco ntuple production, branch contents, ROOT-to-HDF5 conversion, and dataset integration.
- `docs/data.md` lists the Phase-2 SingleMu training sample and per-event track/segment multiplicity statistics.
- `docs/data-format.md` describes model input/output tensor shapes, dtypes, feature order, masks, and preprocessing (reference for ONNX export).
- `docs/metric.md` defines the primary evaluation metric (TNR at TPR >= 99.9%), its `BinarySpecificityAtSensitivity` computation, and the evaluation procedure.
- `docs/onnx.md` describes ONNX export (`scripts/export.py`, `Phase2NoHitModelWrapper`) with preprocessing baked into the graph.
