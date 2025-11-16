"""TurboLoader: High-performance data loading for machine learning

TurboLoader is a C++-based data loading library that achieves 30-35x speedup
over PyTorch DataLoader through SIMD optimizations, lock-free concurrency,
and zero-copy I/O.

Example usage:
    >>> import turboloader
    >>> config = turboloader.Config()
    >>> config.num_workers = 16
    >>> pipeline = turboloader.Pipeline(["data.tar"], config)
    >>> pipeline.start()
    >>> batch = pipeline.next_batch(256)
"""

__version__ = "0.3.2"
__author__ = "Arnav Jain"
__license__ = "MIT"

# Import will happen from the built C++ extension
# which is placed in build/python/ during compilation
try:
    from .turboloader import *  # noqa: F401, F403
except ImportError:
    # During package building, the extension may not be available yet
    pass

