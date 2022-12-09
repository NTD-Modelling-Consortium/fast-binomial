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
    std::optional<CacheArray> cache_;
};

class FastBinomial
{
public:
    using value_type = int;
    using distribution_type = Eigen::Rand::BinomialGen<value_type>;
    using pool_type = RandomPool<value_type, distribution_type>;

    explicit FastBinomial(double p);
    value_type generate(unsigned int n);
    value_type generate(unsigned int n, double p);

private:
    PRNG generator_;
    const double p_;
    std::vector<std::optional<pool_type>> binomials_;
    std::vector<std::unordered_map<double, std::optional<pool_type>>> distributions_;
};

#include "fast_binomial.inl"
