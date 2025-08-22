import torch
import torch.nn as nn
from torch import Tensor





class ToDtype(nn.Module):

    dtype_map = {
        "float": torch.float32,
        "float32": torch.float32,
        "double": torch.float64,
        "float64": torch.float64,
        "long": torch.int64,
        "int64": torch.int64,
        "int": torch.int32,
        "int32": torch.int32,
        "half": torch.float16,
        "float16": torch.float16,
        "bool": torch.bool,
    }

    def __init__(self, dtype: str):
        super().__init__()
        if dtype not in self.dtype_map:
            raise ValueError(f"Unsupported dtype: {dtype}. Supported dtypes are: {list(self.dtype_map.keys())}")
        self.dtype = self.dtype_map[dtype]

    def forward(self, input: Tensor) -> Tensor:
        return input.to(self.dtype)
