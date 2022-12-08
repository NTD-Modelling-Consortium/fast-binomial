
inline BinomialPool::value_type FastBinomial::generate_one(unsigned int n)
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
            binomials.emplace(generator_, BinomialDist(n, p_), block_size_);
        }
        return binomials->next();
    }
}
