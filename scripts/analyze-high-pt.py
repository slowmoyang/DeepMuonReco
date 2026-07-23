"""Offline diagnosis of the high-pT background rejection deficit.

Consumes the flat score table written by ``scripts/dump-scores.py`` and answers,
without retraining, whether the collapse of background rejection at high pT is a
training-exposure problem or an intrinsic ceiling:

D1  within-bin ranking   per-pT-bin ROC AUC and TNR at a per-bin TPR target,
                         which separates ranking quality from where the single
                         global threshold happens to land.
D2  irreducible fraction background composition by truth match. A negative that
                         is a truth-matched simulated muon is geometrically
                         indistinguishable from a positive given the current
                         inputs, so it bounds what any training can achieve.
D3  ghost fraction       dR from each high-pT negative to the nearest positive
                         in the same event. The model has no track-track
                         self-attention, so a duplicate of a positive cannot be
                         separated from it.
D4  prior encoding       median background score against P(signal | pT).
D5  loss exposure        share of the total training loss carried by each pT
                         bin under the run's own criterion.
"""

import argparse
import json
import logging
from pathlib import Path

import h5py as h5
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from rich.logging import RichHandler
from rich_argparse import ArgumentDefaultsRichHelpFormatter
from scipy.stats import rankdata

import matplotlib.pyplot as plt
import mplhep as mh

from muonly.metrics import assign_pt_bins, compute_pt_binned_operating_point
from muonly.nn.metrics import DEFAULT_PT_EDGES, format_pt_bin_labels
from muonly.utils.plot import save_figure

mh.style.use("CMS")

_logger = logging.getLogger(Path(__file__).name)
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()],
)

# Background truth categories, in priority order. Fixed assignment: a category
# keeps its colour regardless of how many categories a given bin happens to have.
CATEGORY_COLORS = {
    "truth_muon": "tab:red",
    "reco_muon_not_truth": "tab:orange",
    "non_muon": "tab:blue",
}


def rank_auc(score: np.ndarray, target: np.ndarray) -> float:
    """ROC AUC via the Mann-Whitney rank statistic (ties averaged)."""
    n_positive = int(target.sum())
    n_negative = target.size - n_positive
    if n_positive == 0 or n_negative == 0:
        return float("nan")
    ranks = rankdata(score)
    return float(
        (ranks[target == 1].sum() - n_positive * (n_positive + 1) / 2)
        / (n_positive * n_negative)
    )


def categorize_background(table: dict[str, np.ndarray], mask: np.ndarray) -> dict:
    """Partition background tracks by how muon-like they are at truth level."""
    is_truth = table["is_matched_muon"][mask] == 1
    is_reco = table["is_reco_muon"][mask] == 1
    total = int(mask.sum())
    counts = {
        "truth_muon": int(is_truth.sum()),
        "reco_muon_not_truth": int((is_reco & ~is_truth).sum()),
        "non_muon": int((~is_reco & ~is_truth).sum()),
    }
    return {
        "n_background": total,
        "counts": counts,
        "fractions": {
            name: (value / total if total else float("nan"))
            for name, value in counts.items()
        },
    }


def nearest_positive_delta_r(
    table: dict[str, np.ndarray],
    query_mask: np.ndarray,
) -> np.ndarray:
    """dR from each queried track to the nearest positive in the same event."""
    event_index = table["event_index"]
    order = np.argsort(event_index, kind="stable")
    if not np.array_equal(order, np.arange(event_index.size)):
        raise RuntimeError("event_index is not sorted; the table is out of order.")

    is_positive = table["target"] == 1
    boundaries = np.searchsorted(
        event_index, np.arange(event_index[-1] + 2), side="left"
    )

    eta = table["eta"]
    phi = table["phi"]
    query_indices = np.flatnonzero(query_mask)
    result = np.full(query_indices.size, np.inf, dtype=np.float64)

    for slot, index in enumerate(query_indices):
        event = event_index[index]
        lo, up = boundaries[event], boundaries[event + 1]
        positives = np.flatnonzero(is_positive[lo:up]) + lo
        if positives.size == 0:
            continue
        delta_eta = eta[positives] - eta[index]
        delta_phi = np.abs(phi[positives] - phi[index])
        delta_phi = np.minimum(delta_phi, 2 * np.pi - delta_phi)
        result[slot] = np.sqrt(delta_eta**2 + delta_phi**2).min()

    return result


def loss_exposure(
    table: dict[str, np.ndarray],
    bin_indices: np.ndarray,
    n_bins: int,
    criterion: torch.nn.Module,
) -> dict:
    """Share of the total per-track loss carried by each pT bin and class."""
    logits = torch.from_numpy(
        np.log(np.clip(table["score"], 1e-7, 1 - 1e-7).astype(np.float64))
        - np.log1p(-np.clip(table["score"], 1e-7, 1 - 1e-7).astype(np.float64))
    ).float()
    target = torch.from_numpy(table["target"].astype(np.float32))

    per_track = np.empty(logits.numel(), dtype=np.float64)
    step = 8_000_000
    with torch.inference_mode():
        for lo in range(0, logits.numel(), step):
            sl = slice(lo, lo + step)
            per_track[sl] = (
                criterion(input=logits[sl], target=target[sl]).double().numpy()
            )

    is_positive = table["target"] == 1
    total = per_track.sum()
    positive_share = np.bincount(
        bin_indices[is_positive], weights=per_track[is_positive], minlength=n_bins
    )
    negative_share = np.bincount(
        bin_indices[~is_positive], weights=per_track[~is_positive], minlength=n_bins
    )
    return {
        "total_loss": float(total),
        "positive_share": (positive_share / total).tolist(),
        "negative_share": (negative_share / total).tolist(),
    }


def run(scores_path: Path, output_dir: Path, per_bin_min_tpr: float) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5.File(scores_path, "r") as file:
        table = {name: file[name][:] for name in file.keys()}
        checkpoint_path = Path(file.attrs["checkpoint"])
    _logger.info(f"Loaded {table['score'].size:,} rows from {scores_path}")

    run_dir = checkpoint_path.parents[1]
    config = OmegaConf.load(run_dir / "config.yaml")
    edges = [float(each) for each in OmegaConf.select(config, "eval.pt_edges")
             or DEFAULT_PT_EDGES]
    labels = format_pt_bin_labels(edges)
    n_bins = len(labels)

    bin_indices = assign_pt_bins(pt=table["pt"], edges=edges)
    is_positive = table["target"] == 1

    report: dict = {
        "scores": str(scores_path),
        "checkpoint": str(checkpoint_path),
        "pt_edges": edges,
        "pt_bin_labels": labels,
    }

    # ---------------------------------------------------------------------------
    # global operating point + per-bin rates (the deployment view)
    # ---------------------------------------------------------------------------
    global_threshold = float(
        np.quantile(table["score"][is_positive], 1.0 - 0.9999, method="lower")
    )
    report["operating_point"] = compute_pt_binned_operating_point(
        y_true=table["target"],
        y_score=table["score"],
        pt=table["pt"],
        edges=edges,
        global_threshold=global_threshold,
        global_min_tpr=0.9999,
        per_bin_min_tpr=(0.999, 0.9999),
    )

    # ---------------------------------------------------------------------------
    # D1 - within-bin ranking quality
    # ---------------------------------------------------------------------------
    _logger.info("D1: per-bin ROC AUC ...")
    auc = []
    for index in range(n_bins):
        in_bin = bin_indices == index
        auc.append(rank_auc(table["score"][in_bin], table["target"][in_bin]))
    report["auc_per_pt_bin"] = auc

    # ---------------------------------------------------------------------------
    # D2 - background composition
    # ---------------------------------------------------------------------------
    _logger.info("D2: background composition ...")
    report["background_composition"] = [
        categorize_background(table, (bin_indices == index) & ~is_positive)
        for index in range(n_bins)
    ]

    # ---------------------------------------------------------------------------
    # D3 - ghost/duplicate fraction among high-pT negatives
    # ---------------------------------------------------------------------------
    high_pt_threshold = 20.0
    query_mask = (~is_positive) & (table["pt"] >= high_pt_threshold)
    _logger.info(f"D3: dR to nearest positive for {int(query_mask.sum()):,} tracks ...")
    delta_r = nearest_positive_delta_r(table, query_mask)
    finite = np.isfinite(delta_r)
    report["ghost_matching"] = {
        "pt_threshold": high_pt_threshold,
        "n_background": int(query_mask.sum()),
        "n_with_positive_in_event": int(finite.sum()),
        "quantiles": {
            str(q): float(np.quantile(delta_r[finite], q))
            for q in (0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9)
        },
        "fraction_within": {
            str(cut): float((delta_r < cut).mean()) for cut in (0.01, 0.05, 0.1, 0.3)
        },
    }

    # ---------------------------------------------------------------------------
    # D4 - how much of the score is the pT prior
    # ---------------------------------------------------------------------------
    _logger.info("D4: score vs pT prior ...")
    median_background_score = []
    signal_prior = []
    for index in range(n_bins):
        in_bin = bin_indices == index
        background = in_bin & ~is_positive
        median_background_score.append(
            float(np.median(table["score"][background])) if background.any() else float("nan")
        )
        n_in_bin = int(in_bin.sum())
        signal_prior.append(
            float((in_bin & is_positive).sum() / n_in_bin) if n_in_bin else float("nan")
        )
    report["median_background_score"] = median_background_score
    report["signal_prior"] = signal_prior

    # ---------------------------------------------------------------------------
    # D5 - loss exposure under the run's own criterion
    # ---------------------------------------------------------------------------
    _logger.info("D5: loss exposure per pT bin ...")
    pos_weight = config.loss.pos_weight
    if not isinstance(pos_weight, (int, float)):
        _logger.warning(
            f"loss.pos_weight={pos_weight!r} is not numeric in the archived config; "
            f"skipping D5."
        )
    else:
        criterion = instantiate(
            config.loss.criterion, pos_weight=torch.tensor(float(pos_weight))
        )
        report["loss_exposure"] = loss_exposure(
            table=table,
            bin_indices=bin_indices,
            n_bins=n_bins,
            criterion=criterion,
        )
        report["loss_exposure"]["pos_weight"] = float(pos_weight)

    with (output_dir / "high-pt-analysis.json").open("w") as stream:
        json.dump(report, stream, indent=4)
    _logger.info(f"Wrote {output_dir / 'high-pt-analysis.json'}")

    make_plots(report, output_dir, per_bin_min_tpr)
    print_summary(report, per_bin_min_tpr)


def _format_pt_ticks(ax, edges: list[float]) -> None:
    """Label an indexed axis with the pT bin boundaries.

    The bins are 13 ordered, unequal-width categories including an infinite
    overflow, so they are plotted at integer positions rather than on a
    continuous pT axis: a log axis cannot show the 0-1 GeV edge, and linear bar
    widths on it misrepresent the geometry. Equal spacing also gives the sparse
    high-pT bins the visual weight the study is about.
    """
    positions = np.arange(len(edges) - 1)
    labels = [
        f"{low:g}-" + ("$\\infty$" if np.isinf(up) else f"{up:g}")
        for low, up in zip(edges[:-1], edges[1:])
    ]
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize="x-small")
    ax.set_xlabel(r"Tracker Track $p_{T}$ [GeV]")


def make_plots(report: dict, output_dir: Path, per_bin_min_tpr: float) -> None:
    edges = report["pt_edges"]
    positions = np.arange(len(edges) - 1)
    key = f"{per_bin_min_tpr:g}".replace(".", "p")
    per_bin = report["operating_point"]["per_bin"]

    # rejection: the single global cut vs. the model's within-bin ranking
    fig, ax = plt.subplots()
    ax.plot(
        positions,
        per_bin["rejection_global"],
        ls="none",
        marker="s",
        color="tab:blue",
        label="Global threshold",
    )
    ax.plot(
        positions,
        per_bin[f"rejection_{key}"],
        ls="none",
        marker="o",
        color="tab:orange",
        label=f"Per-bin threshold ({per_bin_min_tpr:.2%} eff.)",
    )
    _format_pt_ticks(ax, edges)
    ax.set_ylabel(r"Background Rejection Rate, $1 - \epsilon_{bkg}$")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", color="0.9", lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(fontsize="small")
    fig.tight_layout()
    save_figure(fig=fig, path=output_dir / "rejection_global_vs_perbin")

    # D1: within-bin ranking quality
    fig, ax = plt.subplots()
    ax.plot(
        positions,
        report["auc_per_pt_bin"],
        ls="none",
        marker="s",
        color="tab:blue",
    )
    _format_pt_ticks(ax, edges)
    ax.set_ylabel("Within-bin ROC AUC")
    ax.set_ylim(0.5, 1.0)
    ax.grid(axis="y", color="0.9", lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save_figure(fig=fig, path=output_dir / "auc_per_pt_bin")

    # D2: what the background actually is
    fig, ax = plt.subplots()
    bottom = np.zeros(positions.size)
    for name, color in CATEGORY_COLORS.items():
        values = np.nan_to_num(
            np.array(
                [each["fractions"][name] for each in report["background_composition"]]
            )
        )
        ax.bar(
            positions,
            values,
            width=0.82,
            bottom=bottom,
            color=color,
            label=name.replace("_", " "),
        )
        bottom += values
    _format_pt_ticks(ax, edges)
    ax.set_ylabel("Fraction of background tracks")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower left", fontsize="small")
    fig.tight_layout()
    save_figure(fig=fig, path=output_dir / "background_composition_pt")

    # D4: is the score just the prior?
    fig, ax = plt.subplots()
    ax.plot(
        positions,
        report["median_background_score"],
        ls="none",
        marker="s",
        color="tab:blue",
        label="Median background score",
    )
    ax.plot(
        positions,
        report["signal_prior"],
        ls="none",
        marker="o",
        color="tab:orange",
        label=r"$P(\mathrm{signal} \mid p_{T})$",
    )
    ax.axhline(
        report["operating_point"]["global_threshold"],
        color="gray",
        ls="--",
        label="Global threshold",
    )
    _format_pt_ticks(ax, edges)
    ax.set_ylabel("Score")
    ax.set_yscale("log")
    ax.grid(axis="y", color="0.9", lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(fontsize="small")
    fig.tight_layout()
    save_figure(fig=fig, path=output_dir / "score_vs_pt_prior")

    plt.close("all")


def print_summary(report: dict, per_bin_min_tpr: float) -> None:
    labels = report["pt_bin_labels"]
    per_bin = report["operating_point"]["per_bin"]
    key = f"{per_bin_min_tpr:g}".replace(".", "p")
    exposure = report.get("loss_exposure")

    header = (
        f"{'pT bin':>10} {'n_bkg':>10} {'rej@glob':>9} "
        f"{f'rej@bin':>9} {'AUC':>7} {'truth mu':>9} {'medScore':>9} {'P(sig)':>8}"
    )
    if exposure is not None:
        header += f" {'loss neg':>9}"
    print("\n" + header)
    print("-" * len(header))
    for index, label in enumerate(labels):
        line = (
            f"{label:>10} {per_bin['n_background'][index]:>10,} "
            f"{per_bin['rejection_global'][index]:>9.4f} "
            f"{per_bin[f'rejection_{key}'][index]:>9.4f} "
            f"{report['auc_per_pt_bin'][index]:>7.4f} "
            f"{report['background_composition'][index]['fractions']['truth_muon']:>9.3f} "
            f"{report['median_background_score'][index]:>9.4f} "
            f"{report['signal_prior'][index]:>8.3f}"
        )
        if exposure is not None:
            line += f" {exposure['negative_share'][index]:>8.4%}"
        print(line)

    ghost = report["ghost_matching"]
    print(
        f"\nHigh-pT (>{ghost['pt_threshold']:g} GeV) background: "
        f"{ghost['n_background']:,} tracks"
    )
    print(f"  dR to nearest positive, median = {ghost['quantiles']['0.5']:.3f}")
    for cut, fraction in ghost["fraction_within"].items():
        print(f"  fraction with dR < {cut}: {fraction:.3%}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=ArgumentDefaultsRichHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--scores",
        dest="scores_path",
        type=Path,
        required=True,
        help="Score table from scripts/dump-scores.py",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        dest="output_dir",
        type=Path,
        default=None,
        help="Output directory (default: alongside the score table)",
    )
    parser.add_argument(
        "--per-bin-min-tpr",
        dest="per_bin_min_tpr",
        type=float,
        default=0.999,
        help=(
            "Per-bin signal efficiency target for the within-bin diagnostic. "
            "0.9999 rests on a handful of tracks in the sparse bins."
        ),
    )
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = args.scores_path.parent / "high-pt-analysis"

    run(**vars(args))


if __name__ == "__main__":
    main()
