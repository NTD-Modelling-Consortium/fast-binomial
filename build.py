from pybind11.setup_helpers import Pybind11Extension, build_ext


def build(setup_kwargs):
    ext_modules = [
        Pybind11Extension("fast_binomial_cpp", ["src/main.cpp"]),
    ]
    setup_kwargs.update(
        {
            "ext_modules": ext_modules,
            "zip_safe": False,
        }
    )
