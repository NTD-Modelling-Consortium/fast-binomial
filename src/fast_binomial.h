#pragma once

#include <optional>
#include <vector>
#include <type_traits>

#include <EigenRand/EigenRand>
#include <Eigen/Dense>

#include "sfc.h"

using PRNG = sfc64; // Eigen::Rand::Vmt19937_64 (SIMD-ed version of MT19937);

template <typename ScalarType, typename DistributionT, size_t CacheSize = 1024>
class RandomPool
{
public:
    explicit RandomPool(PRNG &generator, DistributionT &&distribution);
    ScalarType next();

private:
    using CacheArray = Eigen::Array<ScalarType, CacheSize, 1>;

    PRNG &generator_;
    DistributionT distribution_;
    // defaults indicates that we have to initialise the cache
    unsigned int next_index_ = CacheSize + 1;
    CacheArray cache_;
};

class FastBinomial
{
public:
    using value_type = int;
    using distribution_type = Eigen::Rand::BinomialGen<value_type>;
    using pool_type = RandomPool<value_type, distribution_type>;

    explicit FastBinomial(float p);
    value_type generate(unsigned int n);

private:
    PRNG generator_;
    const float p_;
    std::vector<std::optional<pool_type>> binomials_;
};

#include "fast_binomial.inl"
