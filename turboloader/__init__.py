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
    # The C++ extension is installed at package root
    import turboloader as _turboloader_ext
    if hasattr(_turboloader_ext, 'DataLoader'):
        DataLoader = _turboloader_ext.DataLoader
        version = _turboloader_ext.version
        features = _turboloader_ext.features
        __all__ = ['DataLoader', 'version', 'features', '__version__']
    else:
        __all__ = ['__version__']
except (ImportError, AttributeError):
    # Fallback for development/documentation builds
    __all__ = ['__version__']
