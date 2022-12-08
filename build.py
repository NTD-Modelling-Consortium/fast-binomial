from pybind11.setup_helpers import Pybind11Extension


def build(setup_kwargs):
    ext_modules = [
        Pybind11Extension(
            "fast_binomial_cpp",
            ["src/main.cpp", "src/fast_binomial.cpp"],
            extra_compile_args=["-O3", "-funroll-loops", "-ffast-math"],
        ),
    ]
    setup_kwargs.update(
        {
            "ext_modules": ext_modules,
            "zip_safe": False,
        }
    )
