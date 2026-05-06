import abc
from torch import Tensor
from torch.nn import Module


__all__ = [
    "Indexible",
]


class Indexible(Module, metaclass=abc.ABCMeta):
    def __init__(self, index: list[int] | None = None) -> None:
        super().__init__()

        self.index = index

    @abc.abstractmethod
    def _forward(self, input: Tensor) -> Tensor: ...

    def forward(self, input: Tensor) -> Tensor:
        """
        Applies a transformation to the specified feature indices of the input tensor.

        Args:
            input (Tensor): Input tensor with the shape of (N, L, D), where N
                is the batch size, L is the sequence length, and D is the feature dimension.
        Returns:
            output (Tensor): Output tensor with the same shape as input, where
                the specified indices are transformed using the transformation defined in the subclass.
        """
        output = input.clone()
        if self.index is None:
            output = self._forward(output)
        else:
            output[..., self.index] = self._forward(output[..., self.index])
        return output
