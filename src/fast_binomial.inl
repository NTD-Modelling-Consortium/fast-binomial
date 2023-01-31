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

template<bool is_scalar_p, unsigned short CacheSize, typename PRNG>
inline FastBinomialFixed<is_scalar_p, CacheSize, PRNG>::FastBinomialFixed(
  p_type&& p,
  std::optional<uint64_t> seed)
  : generator_(seed.value_or(std::random_device()()))
  , p_(std::forward<p_type>(p))
{

  if constexpr (is_scalar_p) {
    pools_ = pools_container_type(1);
  } else {
    pools_ = pools_container_type(p_.size());
  }
}

template<bool is_scalar_p, unsigned short CacheSize, typename PRNG>
inline typename FastBinomialFixed<is_scalar_p, CacheSize, PRNG>::value_type
FastBinomialFixed<is_scalar_p, CacheSize, PRNG>::generate(unsigned int n)
{
  if (n == 0) {
    advance_p_index();
    return 0;
  }

  const auto p = next_p();
  auto& pools_for_p = pools_[p_index_];

  // This is potentially dangerous, but it's a "feature". We don't want to check
  // whether provided `n`s are sensible or not. This would reduce performance.
  if (pools_for_p.size() <= n) {
    pools_for_p.resize(n + 1);
  }

  auto& pool = pools_for_p[n];
  if (!pool) {
    pool.emplace(generator_, distribution_type(n, p));
  }

  advance_p_index();

  return pool->next();
}

template<bool is_scalar_p, unsigned short CacheSize, typename PRNG>
inline double
FastBinomialFixed<is_scalar_p, CacheSize, PRNG>::next_p()
{
  if constexpr (is_scalar_p) {
    return p_;
  } else {
    return p_[p_index_];
  }
}

template<bool is_scalar_p, unsigned short CacheSize, typename PRNG>
inline constexpr void
FastBinomialFixed<is_scalar_p, CacheSize, PRNG>::advance_p_index()
{
  if constexpr (!is_scalar_p) {
    p_index_ = (p_index_ + 1) % p_.size();
  }
}
