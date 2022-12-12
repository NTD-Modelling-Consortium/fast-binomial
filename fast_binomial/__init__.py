from enum import Enum
from typing import Optional, Type, overload

import numpy as np
from numpy.typing import NDArray

from fast_binomial_cpp import (
    FBScalarSFC64Block8,
    FBScalarSFC64Block16,
    FBScalarSFC64Block128,
    FBVectorSFC64Block8,
    FBVectorSFC64Block16,
    FBVectorSFC64Block128,
)


class BitGenerator:
    def __str__(self):
        return self.__class__.__name__


class SFC64(BitGenerator):
    pass


# class MT19937(BitGenerator):
#     pass


def _verify_shapes(n_shape: tuple, p_shape: tuple):
    if len(p_shape) <= len(n_shape):
        compare_section = n_shape[0 : len(p_shape)]
        if compare_section == p_shape:
            return None
    raise ValueError("Shape mismatch, p must be smaller or ")


class BlockSize(Enum):
    small = 8
    medium = 16
    large = 128


class Generator:
    bit_generator: BitGenerator
    scalar_generator: Type[FBScalarSFC64Block8] | Type[FBScalarSFC64Block16] | Type[
        FBScalarSFC64Block128
    ]
    vector_generator: Type[FBVectorSFC64Block8] | Type[FBVectorSFC64Block16] | Type[
        FBVectorSFC64Block128
    ]
    fixed_generator: Optional[
        FBScalarSFC64Block8
        | FBScalarSFC64Block16
        | FBScalarSFC64Block128
        | FBVectorSFC64Block8
        | FBVectorSFC64Block16
        | FBVectorSFC64Block128
    ]
    p_cached_shape: Optional[tuple]

    def __init__(
        self,
        bit_generator: BitGenerator,
        cached_binomial_p: Optional[float | NDArray[np.float_]] = None,
        block_size: BlockSize = BlockSize.small,
    ) -> None:
        """
        A faster generator for binomial distributions, that caches values for fixed probability.

        Args:
            bit_generator (BitGenerator): A Bit Generator provided by fast binomial
            cached_binomial_p (Optional[float  |  NDArray[np.float_]], optional): Probability or probabilities
            to cache. In None case only caches based on given values. Defaults to None.
            block_size (BlockSize, optional): Select from different block sizes. Defaults to BlockSize.small.
        """
        if isinstance(bit_generator, SFC64):
            self.bit_generator = bit_generator

            if block_size == BlockSize.small:
                self.scalar_generator = FBScalarSFC64Block8
                self.vector_generator = FBVectorSFC64Block8
            elif block_size == BlockSize.medium:
                self.scalar_generator = FBScalarSFC64Block16
                self.vector_generator = FBVectorSFC64Block16
            else:
                self.scalar_generator = FBScalarSFC64Block128
                self.vector_generator = FBVectorSFC64Block128

            if cached_binomial_p is None:
                self.fixed_generator = None
                self.p_cached_shape = None
            else:
                if isinstance(cached_binomial_p, float):
                    self.p_cached_shape = None
                    self.fixed_generator = self.scalar_generator(cached_binomial_p)
                else:
                    self.p_cached_shape = cached_binomial_p.shape
                    self.fixed_generator = self.vector_generator(
                        cached_binomial_p.flatten().tolist()
                    )
        else:
            raise ValueError(f"Unsupported bit generator {bit_generator}")

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
                _verify_shapes(n.shape, p.shape)
            elif p is None and self.p_cached_shape is not None:
                # If p is not provided but p cached is array
                _verify_shapes(n.shape, self.p_cached_shape)

        if p is None:
            if self.fixed_generator is None:
                raise ValueError("p is required if not supplied for caching")
            else:
                return self.fixed_generator.generate(n)
        else:
            if isinstance(p, float):
                return self.scalar_generator(p).generate(n)
            else:
                return self.vector_generator(p.flatten().tolist()).generate(n)
