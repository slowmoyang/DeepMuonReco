import torch.nn as nn


__all__ = [
    'MultiLayerPerceptron'
]


class MultiLayerPerceptron(nn.Sequential):

    def __init__(
        self,
        input_dim: int,
        output_dim: int | None = None,
        widening_factor: int = 4,
        bias: bool = True,
        dropout_p: float = 0.,
    ) -> None:
        """
        """
        output_dim = output_dim or input_dim

        hidden_dim = widening_factor * input_dim

        super().__init__(
            nn.Linear(
                in_features=input_dim,
                out_features=hidden_dim,
                bias=bias
            ),
            nn.GELU(),
            nn.Linear(
                in_features=hidden_dim,
                out_features=output_dim,
                bias=bias
            ),
            nn.Dropout(
                p=dropout_p
            )
        )
