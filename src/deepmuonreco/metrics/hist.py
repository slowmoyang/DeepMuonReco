from functools import cached_property
from typing import Any
import torch
from torch import Tensor
from torchmetrics import Metric
import aim


__all__ = [
    'Histogram',
]


class Histogram(Metric):

    """
    TODO:
        - plot
        - density
    """

    histogram: Tensor

    def __init__(
        self,
        range,
        bins: int = 512,
        clamp: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            bins: the number of bins
            range: range
            clamp: a boolean indicating whether entries are clamped or not before filling
        """
        super().__init__(**kwargs)
        assert bins <= 512
        # assert len(output) >= 1

        self.bins = bins
        self.range = tuple(range)
        self.clamp = clamp

        self.add_state(
            name='histogram',
            default=torch.zeros((self.bins, )),
            dist_reduce_fx='sum',
        )

    @cached_property
    def bin_edges(self) -> Tensor:
        return torch.linspace(self.range[0], self.range[1], (self.bins + 1))

    @cached_property
    def bin_centers(self) -> Tensor:
        return (self.bin_edges[:-1] + self.bin_edges[1:]) / 2

    @cached_property
    def bin_width(self) -> float:
        return (self.range[1] - self.range[0]) / self.bins

    @cached_property
    def clamp_range(self) -> tuple[float, float]:
        min, max = self.range
        bin_half_width = (max - min) / self.bins
        return (min + bin_half_width, max - bin_half_width)

    def update(
        self,
        input: Tensor,
        weight: Tensor | None = None,
    ) -> None:
        """
        """
        input = input.cpu()
        if weight is not None:
            weight = weight.cpu()

        input = input.float()
        if self.clamp:
            input = input.clamp(*self.clamp_range)

        histogram = torch.histogram(
            input=input,
            bins=self.bins,
            weight=weight,
            range=self.range,
            density=False,
        )
        self.histogram += histogram.hist.to(self.histogram.device)

    def compute(self):
        hist = self.histogram.cpu()

        return aim.Distribution(
            bin_count=self.bins,
            hist=hist.numpy(),
            bin_range=self.range,
        )
