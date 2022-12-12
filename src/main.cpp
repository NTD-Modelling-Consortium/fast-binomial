#include "fast_binomial.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <string>

namespace py = pybind11;

template<unsigned short CacheSize>
void
bind_scalar_generator(py::module& m, const char* name)
{
  py::class_<FastBinomialFixed<true, CacheSize>>(
    m, name, "Fast generator of number from a binomial distribution")

    .def(py::init<double>(),
         py::arg("p"),
         R"doc(
Create a binomial generator for a given probability.

Args:
    p (float): probability)doc")

    .def("generate",
         py::vectorize(&FastBinomialFixed<true, CacheSize>::generate),
         py::arg("n"),
         R"doc(
Generate numbers from binomial distribution for a given `n`
Yeah
Args:
    n (int/list/np.array): number of trials

Returns:
    int/np.array (dependend on input) of binomials)doc");
}

template<unsigned short CacheSize>
void
bind_vector_generator(py::module& m, const char* name)
{
  py::class_<FastBinomialFixed<false, CacheSize>>(
    m, name, "Fast generator of number from a binomial distribution")

    .def(py::init<std::vector<double>>(),
         py::arg("p"),
         R"doc(
Create a binomial generator for a given vector of probabilities.

Args:
    p (floats): probability)doc")

    .def("generate",
         py::vectorize(&FastBinomialFixed<false, CacheSize>::generate),
         py::arg("n"),
         R"doc(
Generate numbers from binomial distribution for a given `n`

Args:
    n (int/list/np.array): number of trials

Returns:
  int/np.array (dependend on input) of binomials)doc");
}

PYBIND11_MODULE(fast_binomial_cpp, m)
{
  m.doc() = "Fast Binomial implementation with caching";

  bind_scalar_generator<8>(m, "FBScalarSFC64Block8");
  bind_scalar_generator<16>(m, "FBScalarSFC64Block16");
  bind_scalar_generator<128>(m, "FBScalarSFC64Block128");

  bind_vector_generator<8>(m, "FBVectorSFC64Block8");
  bind_vector_generator<16>(m, "FBVectorSFC64Block16");
  bind_vector_generator<128>(m, "FBVectorSFC64Block128");
}
