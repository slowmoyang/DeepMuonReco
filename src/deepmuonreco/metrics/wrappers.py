from typing import Any
from torch import Tensor
from tensordict.utils import _unravel_key_to_tuple
from tensordict import TensorDict
from torchmetrics.wrappers.abstract import WrapperMetric
from torchmetrics.metric import Metric


__all__ = [
    'TensorDictWrapper',
    'MaskingWrapper',
]


class TensorDictWrapper(WrapperMetric):
    """
    """

    def __init__(
        self,
        metric: Metric,
        in_keys: list[str | list[str]],
    ) -> None:
        super().__init__()

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
        tensor_list = []
        for in_key in self.in_keys:
            tensor = input._get_tuple_maybe_non_tensor(key=_unravel_key_to_tuple(in_key), default=None)

            if tensor is None:
                raise RuntimeError(
                    f"Expected input to contain key {in_key} but got {input.keys()}. "
                    "Please check the input TensorDict."
                )
            tensor_list.append(tensor)
        return tuple(tensor_list)

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

    def __init__(self, metric: Metric) -> None:
        super().__init__()
        if not isinstance(metric, Metric):
            raise ValueError(f"Expected argument `metric` to be an instance of `torchmetrics.Metric` but got {metric}")
        self.metric = metric

    def _mask_tensors(self, mask: Tensor, *args: Tensor, **kwargs: Tensor) -> tuple[tuple[Tensor, ...], dict[str, Tensor]]:
        args = tuple([each.masked_select(mask) for each in args])
        kwargs = {key: value.masked_select(mask) for key, value in kwargs.items()}
        return args, kwargs

    def forward(self, mask: Tensor, *args, **kwargs) -> Any:
        args, kwargs = self._mask_tensors(mask, *args, **kwargs)
        return self.metric(*args, **kwargs)

    def update(self, mask: Tensor, *args, **kwargs) -> None:
        args, kwargs = self._mask_tensors(mask, *args, **kwargs)
        return self.metric.update(*args, **kwargs)

    def compute(self) -> Any:
        return self.metric.compute()

    def reset(self) -> None:
        self.metric.reset()
