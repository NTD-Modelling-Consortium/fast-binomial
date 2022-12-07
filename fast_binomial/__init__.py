import numpy as np
from numpy.typing import NDArray

from fast_binomial_cpp import *


class FastBinomial:
    def __init__(self, p: float, block_size: int = 10000) -> None:
        pass

    def __call__(self, arr: NDArray[np.int_] | int) -> NDArray[np.int_] | int:
        raise NotImplementedError
