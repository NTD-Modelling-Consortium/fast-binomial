#include "fast_binomial.h"

#include "sfc.h"
#include <EigenRand/EigenRand>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <optional>
#include <string>

namespace py = pybind11;

template<unsigned short CacheSize, typename PRNG>
void
bind_scalar_generator(py::module& m, const char* name)
{
  py::class_<FastBinomialFixed<true, CacheSize, PRNG>>(
    m, name, "Fast generator of number from a binomial distribution")

    .def(py::init<double, std::optional<uint64_t>>(),
         py::arg("p"),
         py::arg("seed") = py::none(),
         R"doc(
Create a binomial generator for a given probability.

Args:
    p (float): probability)doc")

    .def("generate",
         py::vectorize(&FastBinomialFixed<true, CacheSize, PRNG>::generate),
         py::arg("n"),
         R"doc(
Generate numbers from binomial distribution for a given `n`
Yeah
Args:
    n (int/list/1D np.array): number of trials

Returns:
    int/np.array (dependend on input) of binomials)doc");
}

template<unsigned short CacheSize, typename PRNG>
void
bind_vector_generator(py::module& m, const char* name)
{
  py::class_<FastBinomialFixed<false, CacheSize, PRNG>>(
    m, name, "Fast generator of number from a binomial distribution")

    .def(py::init<std::vector<double>, std::optional<uint64_t>>(),
         py::arg("p"),
         py::arg("seed") = py::none(),
         R"doc(
Create a binomial generator for a given vector of probabilities.

Args:
    p (floats): probability)doc")

    .def("generate",
         py::vectorize(&FastBinomialFixed<false, CacheSize, PRNG>::generate),
         py::arg("n"),
         R"doc(
Generate numbers from binomial distribution for a given `n`

Args:
    n (int/list/1D np.array): number of trials

Returns:
  int/np.array (dependend on input) of binomials)doc");
}

PYBIND11_MODULE(fast_binomial_cpp, m)
{
  m.doc() = "Fast Binomial implementation with caching of vectorised-generated "
            "binomial distribution numbers";

  // SIMD-ed version of MT19937
  using mt19937 = Eigen::Rand::Vmt19937_64;

  bind_scalar_generator<8, sfc64>(m, "FBScalarSFC64");
  bind_vector_generator<8, sfc64>(m, "FBVectorSFC64");
  bind_scalar_generator<8, mt19937>(m, "FBScalarMT19937");
  bind_vector_generator<8, mt19937>(m, "FBVectorMT19937");
}
