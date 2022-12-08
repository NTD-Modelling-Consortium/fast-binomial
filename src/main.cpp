#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "fast_binomial.h"

namespace py = pybind11;

PYBIND11_MODULE(fast_binomial_cpp, m)
{
    m.doc() = "Fast Binomial implementation with caching";

    py::class_<FastBinomial>(m, "FastBinomial")
        .def(py::init<float, unsigned int>())
        .def("generate", py::vectorize(&FastBinomial::generate_one), py::arg("n"));
}
