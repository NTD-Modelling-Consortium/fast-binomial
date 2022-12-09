from typing import Optional, overload

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


def verify_shapes(n_shape: tuple, p_shape: tuple):
    if len(p_shape) <= len(n_shape):
        compare_section = n_shape[0 : len(p_shape)]
        if compare_section == p_shape:
            return None
    raise ValueError("Shape mismatch, p must be smaller or ")


class Generator:
    def __init__(
        self,
        bit_generator: BitGenerator,
        cached_binomial_p: Optional[float | NDArray[np.float_]] = None,
    ) -> None:
        if isinstance(bit_generator, SFC64):
            self.bit_generator = bit_generator
            if cached_binomial_p is None:
                self.fixed_generator = None
                self.p_cached_shape = None
            else:
                if isinstance(cached_binomial_p, float):
                    self.p_cached_shape = None
                    self.fixed_generator = FBScalarFixedSFC64(cached_binomial_p)
                else:
                    self.p_cached_shape = cached_binomial_p.shape
                    flat_p = cached_binomial_p.flatten()
                    self.fixed_generator = FBVectorFixedSFC64(flat_p)
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

    @overload
    def binomial(self, n: int | NDArray[np.int_]) -> int | NDArray[np.int_]:
        """
        Uses a cached value for p to accelerate binomial generation.
        n must match shape of cached p.

        Args:
            n (int | NDArray[np.int_]): n_trials

        Returns:
            int | NDArray[np.int_]: Samples of the binomial distribution
        """
        ...

    @overload
    def binomial(
        self, n: int | NDArray[np.int_], p: float | NDArray[np.float_]
    ) -> int | NDArray[np.int_]:
        """
        Generates binomial samples, accelerated with caching.
        n must match shape of p.

        Args:
            n (int | NDArray[np.int_]): n_trials
            p (float | NDArray[np.float_]): _description_

        Returns:
            int | NDArray[np.int_]: _description_
        """
        ...

    def binomial(
        self, n: int | NDArray[np.int_], p: Optional[float | NDArray[np.float_]] = None
    ) -> int | NDArray[np.int_]:
        if isinstance(n, int):
            if p is None and self.p_cached_shape is not None:
                raise ValueError("Cached p is array but n is int")
            elif p is not None and not isinstance(p, float):
                # If p is not provided and p cached is array or p is provided and is array
                raise ValueError("Provided p is array but n is int")
        else:
            if p is not None and not isinstance(p, float):
                # If p is provided and array
                verify_shapes(n.shape, p.shape)
            elif p is None and self.p_cached_shape is not None:
                # If p is not provided but p cached is array
                verify_shapes(n.shape, self.p_cached_shape)

        if p is None:
            if self.fixed_generator is None:
                raise ValueError("p is required if not supplied for caching")
            else:
                return self.fixed_generator.generate(n)
        else:
            if isinstance(p, float):
                return self.scalar_generator.generate(n, p)
            else:
                return self.vector_generator.generate(n, p)
