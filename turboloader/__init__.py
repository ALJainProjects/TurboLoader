"""TurboLoader: High-performance data loading for machine learning.

v0.4.0 Features:
- Remote TAR support (HTTP, S3, GCS)
- GPU-accelerated JPEG decoding (nvJPEG)
- Lock-free SPSC queues
- 52+ Gbps local file throughput
- Multi-format pipeline (images, video, tabular data)
"""

__version__ = "0.4.0"

# Note: The actual turboloader module is a compiled C++ extension
# that gets built during installation. This is just a stub for packaging.
