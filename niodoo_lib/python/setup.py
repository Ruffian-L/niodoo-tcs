"""
Setup script for NIODOO Python library.
"""

from setuptools import setup, find_packages
from pyo3_build_config import PyO3Configuration

# Try to import pyo3_build_config for Rust extension building
try:
    from pyo3_build_config import build_rust_extension
    HAS_PYO3_BUILD = True
except ImportError:
    HAS_PYO3_BUILD = False

setup(
    name="niodoo",
    version="0.1.0",
    description="NIODOO Python library for sandboxed code execution",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "giotto-tda>=0.6.0",
    ],
    python_requires=">=3.8",
    # Rust extension module (built separately with cargo build --features pyo3)
    # The Python package expects the Rust module to be installed separately
    # or available in PYTHONPATH
)



