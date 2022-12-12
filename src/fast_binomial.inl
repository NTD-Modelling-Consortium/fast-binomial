#include <random>

template<typename ScalarType,
         typename DistributionT,
         typename PRNG,
         unsigned short CacheSize>
RandomPool<ScalarType, DistributionT, PRNG, CacheSize>::RandomPool(
  PRNG& generator,
  DistributionT&& distribution)
  : generator_(generator)
  , distribution_(std::move(distribution))
  , cache_(std::nullopt)
{
}

template<typename ScalarType,
         typename DistributionT,
         typename PRNG,
         unsigned short CacheSize>
ScalarType
RandomPool<ScalarType, DistributionT, PRNG, CacheSize>::next()
{
  if (next_index_ >= CacheSize) [[unlikely]] {
    cache_ =
      distribution_.generateLike(cache_.value_or(CacheArray{}), generator_);
    next_index_ = 0;
  }

  return cache_.value()[next_index_++];
}

template<bool scalar_p, unsigned short CacheSize, typename PRNG>
inline FastBinomialFixed<scalar_p, CacheSize, PRNG>::FastBinomialFixed(
  p_type&& p)
  : generator_(std::random_device()())
  , p_(std::forward<p_type>(p))
{

  if constexpr (scalar_p) {
    pools_ = pools_container_type(1);
  } else {
    pools_ = pools_container_type(p_.size());
  }
}

template<bool scalar_p, unsigned short CacheSize, typename PRNG>
inline FastBinomialFixed<scalar_p, CacheSize, PRNG>::value_type
FastBinomialFixed<scalar_p, CacheSize, PRNG>::generate(unsigned int n)
{
  if (n == 0) {
    // TODO: refactor this, the same thing is on the bottom
    if constexpr (!scalar_p) {
      p_index_ = (p_index_ + 1) % p_.size();
    }
    return 0;
  }

  const auto p = next_p();
  auto& pools_for_p = pools_[p_index_];

  // TODO: this is dangerous. We should limit n
  if (pools_for_p.size() <= n) {
    pools_for_p.resize(n + 1);
  }

  auto& pool = pools_for_p[n];
  if (!pool) {
    pool.emplace(generator_, distribution_type(n, p));
  }

  if constexpr (!scalar_p) {
    p_index_ = (p_index_ + 1) % p_.size();
  }

  return pool->next();
}

template<bool scalar_p, unsigned short CacheSize, typename PRNG>
inline double
FastBinomialFixed<scalar_p, CacheSize, PRNG>::next_p()
{
  if constexpr (scalar_p) {
    return p_;
  } else {
    return p_[p_index_];
  }
}
