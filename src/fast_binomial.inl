#include <random>

template <typename ScalarType, typename DistributionT, size_t CacheSize>
RandomPool<ScalarType, DistributionT, CacheSize>::RandomPool(PRNG &generator, DistributionT &&distribution)
    : generator_(generator), distribution_(std::move(distribution))
{
}

template <typename ScalarType, typename DistributionT, size_t CacheSize>
ScalarType RandomPool<ScalarType, DistributionT, CacheSize>::next()
{
    if (next_index_ >= CacheSize) [[unlikely]]
    {
        cache_ = distribution_.generateLike(cache_, generator_);
        next_index_ = 0;
    }

    return cache_[next_index_++];
}


inline FastBinomial::FastBinomial(float p)
    : generator_(std::random_device()()), p_(p)
{
}

inline FastBinomial::value_type FastBinomial::generate(unsigned int n)
{
    if (n == 0)
    {
        return 0;
    }
    else
    {
        if (binomials_.size() <= n)
        {
            binomials_.resize(n + 1);
        }

        auto &binomials = binomials_[n];
        if (!binomials)
        {
            binomials.emplace(generator_, distribution_type(n, p_));
        }
        return binomials->next();
    }
}
