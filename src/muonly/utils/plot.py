from dataclasses import dataclass
from pathlib import Path
import numpy as np
from hist import Hist
from hist.intervals import clopper_pearson_interval
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes


__all__ = [
    "Efficiency",
]


def save_figure(
    fig: Figure,
    path: Path,
    suffix_list: list[str] | None = None,
) -> None:
    for suffix in suffix_list or ["png"]:
        fig.savefig(path.with_suffix(f".{suffix}"))


@dataclass
class Efficiency:
    x: np.ndarray
    y: np.ndarray
    xerr: np.ndarray
    yerr_low: np.ndarray
    yerr_up: np.ndarray

    @property
    def yerr(self) -> tuple[np.ndarray, np.ndarray]:
        return (self.yerr_low, self.yerr_up)

    @classmethod
    def from_array(
        cls,
        num: np.ndarray,
        den: np.ndarray,
        x: np.ndarray,
        xerr: np.ndarray,
    ) -> "Efficiency":
        y = num / den
        y_low, y_up = clopper_pearson_interval(num=num, denom=den)
        yerr_low = y - y_low
        yerr_up = y_up - y
        return cls(x=x, y=y, xerr=xerr, yerr_low=yerr_low, yerr_up=yerr_up)

    @classmethod
    def from_hist(
        cls,
        h_num: Hist,
        h_den: Hist,
    ) -> "Efficiency":
        x = h_den.axes[0].centers
        xerr = h_den.axes[0].widths / 2
        num = h_num.values()
        den = h_den.values()
        return cls.from_array(num=num, den=den, x=x, xerr=xerr)

    def plot(self, ax: Axes | None = None, **kwargs) -> Figure:
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()
        ax.errorbar(
            x=self.x,
            y=self.y,
            yerr=self.yerr,
            xerr=self.xerr,
            **kwargs,
        )
        return fig

    def to_npz(self, path: Path) -> None:
        np.savez(
            path,
            x=self.x,
            y=self.y,
            xerr=self.xerr,
            yerr_low=self.yerr_low,
            yerr_up=self.yerr_up,
        )

    @classmethod
    def from_npz(cls, path: Path) -> "Efficiency":
        data = np.load(path)
        return cls(
            x=data["x"],
            y=data["y"],
            xerr=data["xerr"],
            yerr_low=data["yerr_low"],
            yerr_up=data["yerr_up"],
        )
