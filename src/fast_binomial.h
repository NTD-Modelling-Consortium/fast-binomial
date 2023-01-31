#pragma once

#include <Eigen/Dense>
#include <EigenRand/EigenRand>

#include <optional>
#include <type_traits>
#include <vector>

/**
 * Pool of random numbers for a given distribution. Generated lazily in batches
 * of CacheSize -- whenever .next() is called.
 *
 * This is faster than generating the numbers one-by-one because, because of
 * the vectorisation of EigenRang generators.
 */
template<typename ScalarType,
         typename DistributionT,
         typename PRNG,
         unsigned short CacheSize>
class RandomPool
{
public:
  explicit RandomPool(PRNG& generator, DistributionT&& distribution);
  ScalarType next();

private:
  using CacheArray = Eigen::Array<ScalarType, CacheSize, 1>;

  PRNG& generator_;
  DistributionT distribution_;
  // defaults indicates that we have to initialise the cache
  unsigned int next_index_ = CacheSize + 1;
  std::optional<CacheArray> cache_;
};

/**
 * Fast binomial implementation using RandomPool. Based on template parameter
 * `is_scalar_p`, it takes either a scalar probability in the constructor, or a
 * vector of probabilities.
 *
 * When vector of probabilities is used, .generate(n) just goes in a cycle
 * over `p`s from the vector. It's up to the client to use `n`s for
 * corresponding `p`s.
 */
template<bool is_scalar_p, unsigned short CacheSize, typename PRNG>
class FastBinomialFixed
{
public:
  using value_type = int;
  using distribution_type = Eigen::Rand::BinomialGen<value_type>;
  using pool_type = RandomPool<value_type, distribution_type, PRNG, CacheSize>;
  // TODO: use np.array instead
  using p_type = std::conditional_t<is_scalar_p, double, std::vector<double>>;

  explicit FastBinomialFixed(p_type&& p, std::optional<uint64_t> seed);
  value_type generate(unsigned int n);

private:
  using pools_container_type =
    std::vector<std::vector<std::optional<pool_type>>>;

  constexpr void advance_p_index();
  double next_p();

  PRNG generator_;
  const p_type p_;
  // it will stay 0 for is_scalar_p, but will keep changing by 1 for non-scalar
  // p in advance_p_index()
  int p_index_ = 0;
  pools_container_type pools_;
};

#include "fast_binomial.inl"
