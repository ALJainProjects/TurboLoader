"""TurboLoader: High-performance data loading for machine learning.

v0.5.2 Features:
- Multi-GPU pipeline support with comprehensive tests
- TensorFlow/Keras integration with test suite
- JAX/Flax integration with device sharding tests
- Distributed training support (multi-node) with tests
- Remote TAR support (HTTP, S3, GCS)
- GPU-accelerated JPEG decoding (nvJPEG)
- Lock-free SPSC queues
- 52+ Gbps local file throughput
- Multi-format pipeline (images, video, tabular data)
- Comprehensive test coverage for all components
"""

__version__ = "0.5.2"

# Import C++ extension module
try:
    from _turboloader import DataLoader, version, features
    __all__ = ['DataLoader', 'version', 'features', '__version__']
except ImportError:
    # Fallback for development/documentation builds
    __all__ = ['__version__']
