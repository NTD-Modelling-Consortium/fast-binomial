#pragma once

#include <vector>
#include <random>
#include <optional>
#include <unordered_map>
#include <type_traits>

#include <pybind11/numpy.h>


using BinomialDist = std::binomial_distribution<unsigned char>;

// template<typename Gen>
// class GeneratorPool
// {
// public:
//     explicit GeneratorPool(Gen &&gen, unsigned int block_size);
//     std::result_of<Gen()> next() const;

// private:
//     Gen gen_;
//     const unsigned int block_size_;
//     unsigned int next_index_ = 0;
//     mutable std::vector<float> cache_;
// };

class BinomialPool
{
public:
    using value_type = BinomialDist::result_type;
    explicit BinomialPool(std::mt19937 &generator, BinomialDist &&distribution, unsigned int block_size);
    value_type next();

private:
    std::mt19937 &generator_;
    BinomialDist distribution_;
    const unsigned int block_size_;
    mutable unsigned int next_index_ = 0;
    mutable std::vector<value_type> cache_;
};

class FastBinomial
{
public:
    explicit FastBinomial(float p, unsigned int block_size = 1000);

    // pybind11::array_t<BinomialPool::value_type> generate(const pybind11::array_t<unsigned int> &ns);

    inline BinomialPool::value_type generate_one(unsigned int n);

private:
    std::mt19937 generator_;
    const float p_;
    const unsigned int block_size_;
    mutable std::vector<std::optional<BinomialPool>> binomials_;
};

#include "fast_binomial.inl"
