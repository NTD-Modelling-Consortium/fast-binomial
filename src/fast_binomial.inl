#include <random>

template <typename ScalarType, typename DistributionT, size_t CacheSize>
RandomPool<ScalarType, DistributionT, CacheSize>::RandomPool(PRNG &generator, DistributionT &&distribution)
    : generator_(generator), distribution_(std::move(distribution)), cache_(std::nullopt)
{
}

template <typename ScalarType, typename DistributionT, size_t CacheSize>
ScalarType RandomPool<ScalarType, DistributionT, CacheSize>::next()
{
    if (next_index_ >= CacheSize) [[unlikely]]
    {
        cache_ = distribution_.generateLike(cache_.value_or(CacheArray{}), generator_);
        next_index_ = 0;
    }

    return cache_.value()[next_index_++];
}

inline FastBinomial::FastBinomial(double p)
    : generator_(std::random_device()()), p_(p)
{
}

inline FastBinomial::value_type FastBinomial::generate(unsigned int n)
{
    if (n == 0 || p_ == 0.0)
    {
        return 0;
    }

    if (p_ == 1.0)
    {
        return n;
    }

    // TODO: this is dangerous. We should either limit n or use an unordered_map (slower)
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

inline FastBinomial::value_type FastBinomial::generate(unsigned int n, double p)
{
    if (n == 0 || p == 0.0)
    {
        return 0;
    }

    if (p == 1.0)
    {
        return n;
    }

    if (distributions_.size() <= n)
    {
        distributions_.resize(n + 1);
    }

    auto &distributions_per_n = distributions_[n];

    auto it = distributions_per_n.find(p);
    if (it == distributions_per_n.end())
    {
        auto pool = pool_type(generator_, distribution_type(n, p));
        const auto [it_p, _] = distributions_per_n.try_emplace(p, std::move(pool));
        it = it_p;
    }

    return it->second.value().next();
}

// inline FastBinomial::value_type FastBinomial::generate(unsigned int n, double p)
// {
//     if (n == 0 || p == 0.0)
//     {
//         return 0;
//     }

//     return distribution_type(n, p)(generator_);
// }
