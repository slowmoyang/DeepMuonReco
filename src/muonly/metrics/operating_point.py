"""Differential (pT-binned) operating-point tables.

Shared by the final evaluation in ``scripts/train.py`` and by the offline
high-pT analysis, so both describe the operating point the same way.

Two distinct quantities are reported per pT bin, and they answer different
questions (see ``docs/metric.md``):

- **at the global threshold** — the differential behaviour of the single
  production cut. This is the quantity that matters for deployment.
- **at a per-bin threshold** — the model's ranking quality *within* the bin,
  independent of where the global cut happens to land. This is a diagnostic
  only; the per-bin thresholds do not describe a single operating point.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np


__all__ = [
    "assign_pt_bins",
    "compute_pt_binned_operating_point",
]


# A tail estimate resting on fewer than this many signal tracks is noise.
MIN_TAIL_SIGNAL_TRACKS = 5


def assign_pt_bins(pt: np.ndarray, edges: Sequence[float]) -> np.ndarray:
    """Index of the left-closed pT bin of each track.

    Tracks below the first edge or above the last fall into the end bins, so
    every track is counted exactly once.
    """
    inner_edges = np.asarray(edges, dtype=np.float64)[1:-1]
    return np.searchsorted(inner_edges, pt, side="right")


def _threshold_at_min_tpr(
    signal_scores: np.ndarray,
    min_tpr: float,
) -> tuple[float, int]:
    """Lowest threshold keeping at least ``min_tpr`` of ``signal_scores``.

    Returns the threshold and the number of signal tracks the requirement
    allows to fail. Selection uses ``score >= threshold``.
    """
    n_signal = signal_scores.size
    n_allowed_failures = int(np.floor(n_signal * (1.0 - min_tpr)))
    if n_signal == 0:
        return float("nan"), 0
    ordered = np.sort(signal_scores)
    return float(ordered[n_allowed_failures]), n_allowed_failures


def compute_pt_binned_operating_point(
    y_true: np.ndarray,
    y_score: np.ndarray,
    pt: np.ndarray,
    edges: Sequence[float],
    global_threshold: float,
    global_min_tpr: float,
    per_bin_min_tpr: Sequence[float] = (0.999, 0.9999),
) -> dict[str, Any]:
    """Per-pT-bin rates at the global threshold, plus per-bin diagnostics.

    ``global_threshold`` is applied with ``score > threshold`` to match the
    ``BinaryROC`` convention used to derive it; per-bin thresholds are applied
    with ``score >= threshold``.
    """
    y_true = np.asarray(y_true).astype(bool)
    y_score = np.asarray(y_score, dtype=np.float64)
    pt = np.asarray(pt, dtype=np.float64)

    bin_indices = assign_pt_bins(pt=pt, edges=edges)
    n_bins = len(edges) - 1

    passes_global = y_score > global_threshold

    per_bin: dict[str, Any] = {
        "n_signal": [],
        "n_background": [],
        "efficiency_global": [],
        "rejection_global": [],
    }
    for min_tpr in per_bin_min_tpr:
        key = f"{min_tpr:g}".replace(".", "p")
        per_bin[f"threshold_{key}"] = []
        per_bin[f"rejection_{key}"] = []
        per_bin[f"low_stat_{key}"] = []

    for index in range(n_bins):
        in_bin = bin_indices == index
        signal = in_bin & y_true
        background = in_bin & ~y_true

        n_signal = int(signal.sum())
        n_background = int(background.sum())
        per_bin["n_signal"].append(n_signal)
        per_bin["n_background"].append(n_background)

        per_bin["efficiency_global"].append(
            float(passes_global[signal].mean()) if n_signal else float("nan")
        )
        per_bin["rejection_global"].append(
            float(1.0 - passes_global[background].mean())
            if n_background
            else float("nan")
        )

        signal_scores = y_score[signal]
        background_scores = y_score[background]
        for min_tpr in per_bin_min_tpr:
            key = f"{min_tpr:g}".replace(".", "p")
            threshold, n_allowed = _threshold_at_min_tpr(
                signal_scores=signal_scores, min_tpr=min_tpr
            )
            per_bin[f"threshold_{key}"].append(threshold)
            per_bin[f"rejection_{key}"].append(
                float((background_scores < threshold).mean())
                if n_background and n_signal
                else float("nan")
            )
            per_bin[f"low_stat_{key}"].append(n_allowed < MIN_TAIL_SIGNAL_TRACKS)

    rejection_global = np.asarray(per_bin["rejection_global"], dtype=np.float64)

    return {
        "bin_edges": [float(each) for each in edges],
        "global_min_tpr": float(global_min_tpr),
        "global_threshold": float(global_threshold),
        "global": {
            "tpr": float(passes_global[y_true].mean()),
            "tnr": float(1.0 - passes_global[~y_true].mean()),
            "pass_fraction": float(passes_global.mean()),
            "tnr_macro_pt": float(np.nanmean(rejection_global)),
        },
        "per_bin": per_bin,
    }
