"""TurboLoader: High-performance data loading for machine learning.

v0.4.0 Features:
- Remote TAR support (HTTP, S3, GCS)
- GPU-accelerated JPEG decoding (nvJPEG)
- Lock-free SPSC queues
- 52+ Gbps local file throughput
- Multi-format pipeline (images, video, tabular data)
"""

__version__ = "0.4.0"

# Import C++ extension module
try:
    from _turboloader import DataLoader, version, features
    __all__ = ['DataLoader', 'version', 'features', '__version__']
except ImportError:
    # Fallback for development/documentation builds
    __all__ = ['__version__']
