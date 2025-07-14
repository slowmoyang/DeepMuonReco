from typing import Any
from torch import Tensor
from tensordict.utils import _unravel_key_to_tuple
from tensordict import TensorDict
from torchmetrics.wrappers.abstract import WrapperMetric
from torchmetrics.metric import Metric

__all__ = [
    'TensorDictMetricWrapper',
    'MaskingWrapper',
]


class TensorDictMetricWrapper(WrapperMetric):
    """
    """

    def __init__(
        self,
        metric: Metric,
        in_keys: list[str | list[str]],
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        if not isinstance(metric, Metric):
            raise ValueError(f"Expected argument `metric` to be an instance of `torchmetrics.Metric` but got {metric}")
        self.metric = metric
        self.in_keys = [
            each if isinstance(each, str) else tuple(each)
            for each in in_keys
        ]


    def _get_tensors(self, input: TensorDict) -> tuple[Tensor, ...]:
        """
        # https://github.com/pytorch/tensordict/blob/v0.7.2/tensordict/nn/common.py#L1136-L1141
        """
        tensors = tuple(
            input._get_tuple_maybe_non_tensor(
                key=_unravel_key_to_tuple(in_key),
                default=None
            )
            for in_key in self.in_keys
        )
        return tensors # type: ignore

    def forward(self, input: TensorDict) -> Any:
        tensors = self._get_tensors(input)
        return self.metric(*tensors)

    def update(self, input: TensorDict) -> None:
        tensors = self._get_tensors(input)
        return self.metric.update(*tensors)

    def compute(self) -> Any:
        return self.metric.compute()

    def reset(self) -> None:
        self.metric.reset()


class MaskingWrapper(WrapperMetric):
    """
    """

    def __init__(
        self,
        metric: Metric,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        if not isinstance(metric, Metric):
            raise ValueError(f"Expected argument `metric` to be an instance of `torchmetrics.Metric` but got {metric}")
        self.metric = metric


    def _mask(self, *args: Tensor, **kwargs: Tensor):
        data_mask = args[0]
        args = tuple([each[data_mask] for each in args[1:]])
        kwargs = {k: v[data_mask] for k, v in kwargs.items()}
        return args, kwargs

    def forward(self, *args: Tensor, **kwargs: Tensor) -> Any:
        args, kwargs = self._mask(*args, **kwargs)
        return self.metric(*args, **kwargs)

    def update(self, *args: Tensor, **kwargs: Tensor) -> None:
        args, kwargs = self._mask(*args, **kwargs)
        return self.metric.update(*args, **kwargs)

    def compute(self) -> Any:
        return self.metric.compute()

    def reset(self) -> None:
        self.metric.reset()
