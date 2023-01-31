from setuptools import setup
import zipfile
from pathlib import Path

import requests
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
    VENDOR_DIR.mkdir(exist_ok=True)

    vendor_libs = [Eigen(), EigenRand()]
    for lib in vendor_libs:
        lib.fetch()

    extension = Pybind11Extension(
        "fast_binomial_cpp",
        ["src/main.cpp"],
        include_dirs=[VENDOR_DIR.name, *(lib.path() for lib in vendor_libs)],
    )
    extension.cxx_std = 17

    setup_kwargs.update(
        {
            "ext_modules": [extension],
            "zip_safe": False,
        }
    )


packages = ["fast_binomial"]

package_data = {"": ["*"]}

install_requires = [
    "numpy>=1.23.5,<2.0.0",
    "pybind11>=2.10.1,<3.0.0",
    "requests>=2.28.1,<3.0.0",
]

setup_kwargs = {
    "name": "fast-binomial",
    "version": "0.1.0",
    "description": "",
    "long_description": "None",
    "author": "mark-todd",
    "author_email": "markpeter.todd@hotmail.co.uk",
    "maintainer": "None",
    "maintainer_email": "None",
    "url": "None",
    "packages": packages,
    "package_data": package_data,
    "install_requires": install_requires,
    "python_requires": ">=3.10,<4.0",
}

build(setup_kwargs)

setup(**setup_kwargs)
