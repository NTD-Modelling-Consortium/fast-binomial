from pathlib import Path
import requests
import zipfile

from pybind11.setup_helpers import Pybind11Extension


VENDOR_DIR = Path(__file__).resolve().parent / "vendor"


class VendorLib:
    VERSION = ""
    URL = ""
    ZIP_FILE = ""
    UNZIPPED_DIR = ""
    TARGET_DIR = Path("")

    def fetch(self):
        assert (
            self.VERSION
            and self.URL
            and self.ZIP_FILE
            and self.UNZIPPED_DIR
            and self.TARGET_DIR
        )

        if self.TARGET_DIR.exists():
            return self.TARGET_DIR.name

        download_target_file = VENDOR_DIR / self.ZIP_FILE

        # Save eigen in vendor/
        response = requests.get(self.URL, stream=True)
        with download_target_file.open("wb") as ofs:
            for chunk in response.iter_content(chunk_size=1024):
                ofs.write(chunk)

        # extract
        with zipfile.ZipFile(download_target_file) as ifs:
            ifs.extractall(path=self.TARGET_DIR)

    def path(self) -> str:
        return str(self.TARGET_DIR / self.UNZIPPED_DIR)


class Eigen(VendorLib):
    VERSION = "3.4.0"
    ZIP_FILE = f"eigen-{VERSION}.zip"
    URL = f"https://gitlab.com/libeigen/eigen/-/archive/{VERSION}/{ZIP_FILE}"
    TARGET_DIR = VENDOR_DIR / "eigen"
    UNZIPPED_DIR = f"eigen-{VERSION}"


class EigenRand(VendorLib):
    VERSION = "0.4.1"
    ZIP_FILE = f"v{VERSION}.zip"
    URL = f"https://github.com/bab2min/EigenRand/archive/refs/tags/{ZIP_FILE}"
    TARGET_DIR = VENDOR_DIR / "eigen_rand"
    UNZIPPED_DIR = f"EigenRand-{VERSION}/"


def build(setup_kwargs):
    vendor_libs = [Eigen(), EigenRand()]
    for lib in vendor_libs:
        lib.fetch()

    ext_modules = [
        Pybind11Extension(
            "fast_binomial_cpp",
            ["src/main.cpp", "src/fast_binomial.cpp"],
            extra_compile_args=[
                "-Ofast",
                "-funroll-loops",
                "-march=native",
                "--std=c++20",
                "-Wno-unused-local-typedefs",  # Eigen Rand :()
            ],
            include_dirs=[VENDOR_DIR.name, *(lib.path() for lib in vendor_libs)],
        ),
    ]
    setup_kwargs.update(
        {
            "ext_modules": ext_modules,
            "zip_safe": False,
        }
    )
