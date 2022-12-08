#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

#include "fast_binomial.h"

namespace py = pybind11;

PYBIND11_MODULE(fast_binomial_cpp, m)
{
    m.doc() = "Fast Binomial implementation with caching";

    py::class_<FastBinomial>(
        m,
        "FastBinomial",
        "Fast generator of number from a binomial distribution")

        .def(
            py::init<float>(),
            py::arg("p"),
            R"doc(
Create a FastBinomial generator for a given probability and block_size cache.

Args:
    p (int): probability)doc")

        .def(
            "generate",
            py::vectorize(&FastBinomial::generate),
            py::arg("n"),
            R"doc(
Generate numbers from binomial distribution for a given `n`

Args:
    n (int/list/np.array): n - number of trials

Returns:
    int/np.array (dependend on input) of binomials)doc");
}
