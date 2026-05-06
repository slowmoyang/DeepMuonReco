from torch import Tensor
from .index import Indexible


__all__ = [
    # functional
    "signed_log1p",
    "signed_expm1",
    # modules
    "Log",
    "Log1p",
    "SignedLog1p",
    "SignedExpm1",
]


def signed_log1p(x: Tensor):
    return x.sign() * x.abs().log1p()


def signed_expm1(x: Tensor):
    return x.sign() * x.abs().expm1()


class Log(Indexible):
    def _forward(self, input: Tensor) -> Tensor:
        return input.log_()


class Log1p(Indexible):
    def _forward(self, input: Tensor) -> Tensor:
        return input.log1p_()


class SignedLog1p(Indexible):
    def _forward(self, input: Tensor) -> Tensor:
        return signed_log1p(input)


class SignedExpm1(Indexible):
    def _forward(self, input: Tensor) -> Tensor:
        return signed_expm1(input)
