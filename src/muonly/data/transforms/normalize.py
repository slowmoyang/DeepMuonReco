import torch
from torch import nn, Tensor


__all__ = [
    "Normalize",
    "MinMaxScaling",
]


class Normalize(nn.Module):
    """torchvision-like Normalize"""

    offset: Tensor
    scale: Tensor

    def __init__(
        self,
        offset,
        scale,
    ) -> None:
        super().__init__()
        offset = torch.tensor(offset, dtype=torch.float32)
        scale = torch.tensor(scale, dtype=torch.float32)

        self.register_buffer("offset", offset)
        self.register_buffer("scale", scale)

    def forward(self, input: Tensor) -> Tensor:
        # Check if input is empty
        if input.numel() == 0:
            return input
        return (input - self.offset) / self.scale


class MinMaxScaling(nn.Module):
    """Min-Max scaling transformation"""

    input_min: Tensor
    input_max: Tensor
    output_min: Tensor
    output_max: Tensor

    def __init__(
        self,
        input_min,
        input_max,
        output_min: float = 0.0,
        output_max: float = 1.0,
    ) -> None:
        """
        Args:
            input_range: (min, max) tuple specifying the input value range
            output_range: (min, max) tuple specifying the output value range, default is (0, 1)
        """
        super().__init__()

        self.register_buffer(
            name="input_min",
            tensor=torch.tensor(data=input_min, dtype=torch.float),
        )
        self.register_buffer(
            name="input_max",
            tensor=torch.tensor(data=input_max, dtype=torch.float),
        )
        self.register_buffer(
            name="output_min",
            tensor=torch.tensor(data=output_min, dtype=torch.float),
        )
        self.register_buffer(
            name="output_max",
            tensor=torch.tensor(data=output_max, dtype=torch.float),
        )

    def forward(self, input: Tensor) -> Tensor:
        # Scale from input_range to [0, 1], then to output_range
        normalized = (input - self.input_min) / (self.input_max - self.input_min)
        return normalized * (self.output_max - self.output_min) + self.output_min

    def extra_repr(self) -> str:
        return (
            f"input_min={self.input_min.tolist()}, input_max={self.input_max.tolist()}, "
            f"output_min={self.output_min.item()}, output_max={self.output_max.item()}"
        )
