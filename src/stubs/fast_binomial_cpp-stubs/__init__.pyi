"""Fast Binomial implementation with caching"""
from __future__ import annotations

import typing

import numpy

import fast_binomial_cpp

_Shape = typing.Tuple[int, ...]

__all__ = ["FastBinomial"]

class FastBinomial:
    def __init__(self, arg0: float, arg1: int) -> None: ...
    def generate(self, n: numpy.ndarray[numpy.uint32]) -> object: ...
    pass
