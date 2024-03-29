from typing import Optional, Type, overload

import numpy as np
from numpy.typing import NDArray

from fast_binomial_cpp import (
    FBScalarMT19937,
    FBScalarSFC64,
    FBVectorMT19937,
    FBVectorSFC64,
)


class BitGenerator:
    seed: Optional[int]

    def __init__(self, seed: Optional[int] = None) -> None:
        self.seed = seed

    def __str__(self):
        return self.__class__.__name__


class SFC64(BitGenerator):
    pass


class MT19937(BitGenerator):
    pass


def _verify_shapes(n_shape: tuple, p_shape: tuple):
    if len(p_shape) <= len(n_shape):
        compare_section = n_shape[-len(p_shape) :]
        if compare_section == p_shape:
            return None
    raise ValueError("Shape mismatch, p must be smaller or ")


class Generator:
    seed: Optional[int]
    scalar_generator: Type[FBScalarSFC64] | Type[FBScalarMT19937]
    vector_generator: Type[FBVectorSFC64] | Type[FBVectorMT19937]
    fixed_generator: Optional[
        FBScalarSFC64 | FBScalarMT19937 | FBVectorSFC64 | FBVectorMT19937
    ]
    p_cached_shape: Optional[tuple]

    def __init__(
        self,
        bit_generator: BitGenerator,
        cached_binomial_p: Optional[float | NDArray[np.float_]] = None,
    ) -> None:
        """
        A faster generator for binomial distributions, that caches values for fixed probability.

        Args:
            bit_generator (BitGenerator): A Bit Generator provided by fast binomial
            cached_binomial_p (Optional[float  |  NDArray[np.float_]], optional): Probability or probabilities
            to cache. In None case only caches based on given values. Defaults to None.
        """
        if isinstance(bit_generator, SFC64):
            self.scalar_generator = FBScalarSFC64
            self.vector_generator = FBVectorSFC64
        elif isinstance(bit_generator, MT19937):
            self.scalar_generator = FBScalarMT19937
            self.vector_generator = FBVectorMT19937
        else:
            raise ValueError(f"Unsupported bit generator {bit_generator}")
        self.seed = bit_generator.seed

        if cached_binomial_p is None:
            self.fixed_generator = None
            self.p_cached_shape = None
        else:
            if isinstance(cached_binomial_p, float):
                self.p_cached_shape = None
                self.fixed_generator = self.scalar_generator(
                    cached_binomial_p, seed=self.seed
                )
            else:
                self.p_cached_shape = cached_binomial_p.shape
                self.fixed_generator = self.vector_generator(
                    cached_binomial_p.flatten().tolist(), seed=self.seed
                )

    @overload
    def binomial(self, n: int) -> int:
        """
        Uses a cached value for p to accelerate binomial generation.

        Args:
            n (int): n_trials

        Returns:
            int: A sample of the binomial distribution
        """
        ...

    @overload
    def binomial(self, n: NDArray[np.int_]) -> NDArray[np.int_]:
        """
        Uses a cached value for p to accelerate binomial generation.
        n must match shape of cached p, or cached p must be a float.

        Args:
            n (NDArray[np.int_]): n_trials

        Returns:
            NDArray[np.int_]: Samples of the binomial distribution
        """
        ...

    @overload
    def binomial(self, n: int, p: float) -> int:
        """
        Generates binomial samples, accelerated with caching.
        n must match shape of p.

        Args:
            n (int): n_trials
            p (float): Probability

        Returns:
            int: A sample of the binomial distribution
        """
        ...

    @overload
    def binomial(
        self, n: NDArray[np.int_], p: float | NDArray[np.float_]
    ) -> NDArray[np.int_]:
        """
        Generates binomial samples, accelerated with caching.
        n must match shape of p, or p must be a float.

        Args:
            n (NDArray[np.int_]): n_trials
            p (float | NDArray[np.float_]): Probability or probabilities

        Returns:
            NDArray[np.int_]: Samples of the binomial distribution
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
                _verify_shapes(n.shape, p.shape)
            elif p is None and self.p_cached_shape is not None:
                # If p is not provided but p cached is array
                _verify_shapes(n.shape, self.p_cached_shape)

        if p is None:
            if self.fixed_generator is None:
                raise ValueError("p is required if not supplied for caching")
            else:
                if self.p_cached_shape is not None:
                    # p is vector cached, n is array
                    assert not isinstance(n, int)
                    return self.fixed_generator.generate(n.flatten()).reshape(n.shape)
                else:
                    # p is scalar cached
                    return self.fixed_generator.generate(n)
        else:
            if isinstance(p, float):
                return self.scalar_generator(p, seed=self.seed).generate(n)
            else:
                # n is array
                assert not isinstance(n, int)
                return (
                    self.vector_generator(p.flatten().tolist(), seed=self.seed)
                    .generate(n.flatten())
                    .reshape(n.shape)
                )
