#include <algorithm>
#include <iostream>
#include <iterator>

#include "fast_binomial.h"

namespace py = pybind11;

BinomialPool::BinomialPool(std::mt19937 &generator, BinomialDist &&distribution, unsigned int block_size)
    : generator_(generator), distribution_(std::move(distribution)), block_size_(block_size)
{
}

BinomialPool::value_type BinomialPool::next()
{
    if (next_index_ >= block_size_ || cache_.size() == 0) [[unlikely]]
    {
        if (cache_.size() == 0)
        {
            cache_.resize(block_size_);
        }

        std::generate_n(
            cache_.begin(),
            block_size_,
            [this]() { return distribution_(generator_); }
        );
        next_index_ = 0;
    }

    return cache_[next_index_++];
}

FastBinomial::FastBinomial(float p, unsigned int block_size)
    : generator_(std::random_device()()), p_(p), block_size_(block_size)
{
}

// py::array_t<BinomialPool::value_type> FastBinomial::generate(const py::array_t<unsigned int> &ns)
// {
//     auto data = ns.data();
//     auto ret = py::array_t<BinomialPool::value_type>(ns.size());
//     auto mutable_ret = ret.mutable_data();

//     for (int i = 0; i < ns.size(); ++i)
//     {
//         const auto n = data[i];
//         if (n == 0)
//         {
//             mutable_ret[i] = 0;    
//         }
//         else
//         {
//             if (binomials_.size() <= n)
//             {
//                 binomials_.resize(n + 1);
//             }

//             auto &binomials = binomials_[n];
//             if (!binomials)
//             {
//                 binomials.emplace(generator_, BinomialDist(n, p_), block_size_);
//             }
//             mutable_ret[i] = binomials->next();
//         }

//     }

//     return ret;
// }
