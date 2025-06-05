"""
RDB2G-Bench: A comprehensive benchmark for automatic graph modeling of relational databases.

This package provides tools and benchmarks for converting relational database data into graphs
and evaluating the performance of various graph-based analysis methods.
"""

__version__ = "0.1.0"
__author__ = "Dongwon Choi"
__email__ = "chlehdwon@kaist.ac.kr"

# Import main modules for easy access
from . import benchmark
from . import common
from . import dataset

__all__ = [
    "__version__",
    "__author__", 
    "__email__",
    "benchmark",
    "common", 
    "dataset",
] 