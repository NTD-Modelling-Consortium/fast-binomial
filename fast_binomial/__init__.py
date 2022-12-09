from typing import Optional

import numpy as np
from numpy.typing import NDArray

from fast_binomial_cpp import (
    FBScalarDynamicSFC64,
    FBScalarFixedSFC64,
    FBVectorDynamicSFC64,
    FBVectorFixedSFC64,
)


class BitGenerator:
    def __str__(self):
        return self.__class__.__name__


class SFC64(BitGenerator):
    pass


# class MT19937(BitGenerator):
#     pass


class _BaseGenerator:
    pass


class _GeneratorFixed(_BaseGenerator):
    def __init__(
        self, bit_generator: BitGenerator, cached_binomial_p: float | NDArray[np.float_]
    ) -> None:
        if isinstance(bit_generator, SFC64):
            if isinstance(cached_binomial_p, float):
                self._generator = FBScalarFixedSFC64(cached_binomial_p)
            else:
                self._generator = FBVectorFixedSFC64(cached_binomial_p)
        else:
            raise ValueError(f"Unsupported bit generator {bit_generator}")

    def binomial(self, n: int | NDArray[np.int_]) -> int | NDArray[np.int_]:
        return self._generator.generate(n)


class _GeneratorDynamic(_BaseGenerator):
    def __init__(self, bit_generator: BitGenerator) -> None:
        if isinstance(bit_generator, SFC64):
            self.bit_generator = bit_generator
        else:
            raise ValueError(f"Unsupported bit generator {bit_generator}")
        self._scalar_generator = None
        self._vector_generator = None

    @property
    def scalar_generator(self):
        if self._scalar_generator is None:
            if isinstance(self.bit_generator, SFC64):
                self._scalar_generator = FBScalarDynamicSFC64()
            else:
                assert False
        return self._scalar_generator

    @property
    def vector_generator(self):
        if self._vector_generator is None:
            if isinstance(self.bit_generator, SFC64):
                self._vector_generator = FBVectorDynamicSFC64()
            else:
                assert False
        return self._vector_generator

    def binomial(
        self, n: int | NDArray[np.int_], p: float | NDArray[np.float_]
    ) -> int | NDArray[np.int_]:
        if isinstance(p, float):
            return self.scalar_generator.generate(n, p)
        else:
            return self.vector_generator.generate(n, p)


def Generator(
    bit_generator: BitGenerator,
    cached_binomial_p: Optional[float | NDArray[np.float_]] = None,
) -> _BaseGenerator:
    if cached_binomial_p is None:
        return _GeneratorDynamic(bit_generator)
    else:
        return _GeneratorFixed(bit_generator, cached_binomial_p)
