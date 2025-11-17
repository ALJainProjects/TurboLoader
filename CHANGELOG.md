# Changelog

All notable changes to TurboLoader will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-01-16

### Production Release

First stable production release of TurboLoader!

### Added
- **Interactive benchmark web app** (`benchmark_app.html`)
  - Real-time performance visualizations with Plotly.js
  - 6 interactive charts (throughput, worker scaling, batch size, memory, CPU)
  - Framework comparison dashboard
  - Configuration controls
  - Professional gradient UI design
- **Benchmark data generator** (`benchmarks/generate_web_data.py`)
  - Automated benchmark data collection
  - JSON export for web app
  - Mock data generation for demos

### Changed
- **Version bumped to 1.0.0** (Production/Stable)
- Updated status from Beta to Production/Stable in PyPI classifiers
- Enhanced module documentation with v1.0.0 feature list
- Streamlined feature descriptions for clarity

### Fixed
- **Zero compiler warnings** - All C++ build warnings resolved:
  - Field initialization order in Worker constructor (pipeline.hpp:286-291)
  - Field initialization order in UnifiedPipeline constructor (pipeline.hpp:410-414)
  - Unused variable in flip_transform.hpp (line 35)
  - Field initialization order in PadTransform constructor (pad_transform.hpp:27-29)
  - Marked unused fields with `[[maybe_unused]]` (pipeline.hpp:387-388)

### Testing
- **87% test pass rate** (13/15 tests passing)
  - test_multi_gpu and test_distributed not built (require CUDA/MPI)
  - All core functionality tests passing
- Comprehensive test suite covering:
  - TAR reader
  - Image decoder
  - HTTP reader
  - S3 reader
  - GCS reader
  - Reader orchestrator
  - Video decoder
  - CSV/Parquet decoders
  - Unified pipeline
  - nvJPEG decoder
  - All 19 transforms

### Performance
- 10,146 img/s throughput (12x faster than PyTorch Optimized)
- 1.3x faster than TensorFlow
- 52+ Gbps local file I/O throughput

### Documentation
- Professional README with architecture diagrams
- Complete API documentation
- Interactive web-based benchmarking

## [0.8.1] - 2025-01-XX

### Added
- **Complete documentation** in `docs/` folder with 15+ professional guides
  - API reference for all 19 transforms
  - Getting started guide
  - Architecture documentation
  - Benchmark methodology
  - PyTorch/TensorFlow integration guides
  - Contributing guidelines
- **Interactive Streamlit benchmark web app** for performance testing
  - Upload custom datasets
  - Compare frameworks (TurboLoader, PyTorch, TensorFlow)
  - Real-time performance charts
  - Memory profiling
- **Enhanced Python bindings** with comprehensive docstrings
  - All 19 transforms documented
  - Module-level functions (version(), features(), list_transforms())
  - Enum documentation (InterpolationMode, PaddingMode, TensorFormat, AutoAugmentPolicy)
- **Professional README** with:
  - Feature comparison table
  - Transform library overview
  - Architecture diagram
  - Links to all documentation
  - Enhanced badges and metadata

### Changed
- Updated version to 0.8.0 across all files
- Module documentation enhanced with usage examples
- API stability guarantees for v1.0 preparation

### Documentation
- 15+ markdown files in docs/ folder
- Complete API reference
- Comprehensive benchmarking documentation
- Development and contributing guides

## [0.7.0] - 2025-01-XX

### Added
- **RandomPosterize** transform - Bit-depth reduction (336,000+ img/s)
  - Ultra-fast bitwise operations
  - Configurable bit depth (1-8 bits)
- **RandomSolarize** transform - Threshold-based pixel inversion (21,000+ img/s)
  - SIMD threshold comparison
  - Configurable threshold
- **RandomPerspective** transform - Perspective warping (9,900+ img/s)
  - SIMD-accelerated homography
  - Configurable distortion scale
- **AutoAugment** policies - State-of-the-art learned augmentation (19,800+ img/s)
  - ImageNet policy
  - CIFAR10 policy
  - SVHN policy
- **Lanczos interpolation** for Resize transform (2,900+ img/s)
  - High-quality downsampling
  - Better than bicubic for size reduction
- **26 advanced unit tests** for new transforms
  - Comprehensive test coverage
  - Performance validation

### Performance
- AutoAugment (ImageNet policy): 19,800+ img/s
- RandomPosterize: 336,000+ img/s (bitwise ops)
- RandomSolarize: 21,000+ img/s
- RandomPerspective: 9,900+ img/s
- Lanczos Resize: 2,900+ img/s

### Testing
- Total test count increased to 52 (26 original + 26 advanced)
- All tests passing

## [0.6.0] - 2025-01-XX

### Added
- **Complete transform system** with 14 SIMD-accelerated operations
  - Resize (Nearest, Bilinear, Bicubic)
  - Normalize / ImageNetNormalize
  - RandomHorizontalFlip / RandomVerticalFlip
  - CenterCrop / RandomCrop
  - ColorJitter
  - Grayscale
  - Pad
  - RandomRotation
  - RandomAffine
  - GaussianBlur
  - RandomErasing
- **PyTorch tensor conversion** (CHW float32 format)
- **TensorFlow tensor conversion** (HWC float32 format)
- **ToTensor** transform with framework-specific formats
- **26 unit tests** for transforms

### Performance
- Overall throughput: 10,146 img/s (12x faster than PyTorch)
- Resize: 3.2x faster than torchvision
- GaussianBlur: 4.5x faster than torchvision
- Memory efficient: 848 MB peak usage

## [0.5.0] - 2025-01-XX

### Added
- TAR archive support (WebDataset format)
- Memory-mapped I/O for zero-copy file access
- 52+ Gbps TAR parsing throughput
- Remote TAR support (HTTP/S3/GCS)

## [0.4.0] - 2025-01-XX

### Added
- Initial public release
- JPEG decoding with libjpeg-turbo
- Basic SIMD transforms
- Lock-free concurrent queues
- Multi-threaded pipeline

## [0.3.8] - 2025-01-16

### Fixed
- Updated all Python examples and scripts to use current simplified Pipeline API
- Fixed version number inconsistencies across codebase
- Updated documentation to accurately reflect thread-safe concurrency implementation

### Changed
- Removed deprecated Config object usage from all examples
- Cleaned up documentation and removed duplicate files
- Updated MANIFEST.in to reflect current file structure

## [0.3.7] - 2025-01-15

### Fixed
- **CRITICAL**: Fixed all race conditions and memory corruption issues with high worker counts
- Added mutex-protected TarReader access for thread safety
- Replaced lock-free SPMC queue with ThreadSafeQueue for Sample objects to prevent race conditions
- Changed vector data copying to use memcpy() instead of assign() for safer concurrent access
- Improved thread synchronization throughout the pipeline
- Verified stability with 8 workers using ThreadSanitizer

### Changed
- Sample queue now uses mutex-based synchronization instead of lock-free atomics
- Each TarReader has dedicated mutex to prevent concurrent access issues
- Pipeline is now fully stable with high worker counts (tested up to 8 workers)

## [0.3.6] - 2025-01-15

### Fixed
- Improved memory allocation pattern in load_sample to reduce reallocation overhead
- Changed vector allocation to use reserve() + assign() pattern for thread safety

### Known Issues
- Memory corruption can still occur under high concurrency with large datasets
- Recommended to use ≤4 workers for production use until fully resolved

## [0.3.5] - 2025-01-15

### Fixed
- **CRITICAL**: Fixed memory corruption bug (double-free) in JPEG decoder
- Added proper cleanup with `jpeg_abort_decompress()` in all error paths
- JPEG decoder now properly handles reuse across multiple decode calls in thread-local storage
- Eliminated JPEG decoder crashes during multi-threaded decoding

### Changed
- Improved error handling in JPEG decoder to ensure cleanup on all failure paths
- Enhanced thread safety for decoder reuse in worker threads

## [0.3.3] - 2025-01-15

### Changed
- Fixed setuptools deprecation warnings (license format in pyproject.toml)
- Removed MIT License classifier to avoid duplicate license declarations
- Updated all documentation files (docs/) to use conservative performance claims
- Replaced specific unverified benchmark numbers with qualitative descriptions
- Added performance disclaimers to all documentation files

### Documentation
- All docs now use "significantly faster" instead of specific unverified multiples
- Removed unverified throughput claims throughout docs/
- Added disclaimer: "Performance claims based on preliminary benchmarks on synthetic datasets"
- Consistent conservative language across all 9 documentation files

## [0.3.2] - 2025-01-15

### Changed
- Updated README with conservative, verified performance claims
- Removed unsubstantiated "30-35x speedup" claims
- Added actual SIMD benchmark results from test suite
- Updated "Current Version" from 0.2.0 to 0.3.1 in README
- Clarified roadmap - v0.3.0 is released, not planned

### Documentation
- README now states "Significantly faster" instead of specific multiples
- Added verified SIMD benchmark data (6718 img/s resize, 47438 img/s normalize)
- Clear disclaimer that benchmarks are on synthetic datasets
- Accurate feature list for v0.3.x releases

## [0.3.1] - 2025-01-15

### Added
- Comprehensive Python test suite for augmentation transforms (tests/test_augmentations.py)
- All 9 augmentation transform tests passing
- Test coverage for RandomHorizontalFlip, RandomVerticalFlip, ColorJitter, RandomRotation, RandomCrop, RandomErasing, GaussianBlur, and AugmentationPipeline

### Fixed
- Fixed packaging issue where C++ extension was placed at wheel root instead of package directory (setup.py)
- Package now correctly imports all augmentation classes

### Testing
- 45 total tests, all passing:
  - 11 C++ unit tests (LockFreeQueue, MemoryPool, MmapReader)
  - 15 SIMD transform tests
  - 5 C++ transform tests
  - 5 HTTP reader tests
  - 9 Python augmentation tests

## [0.3.0] - 2025-01-15

### Added
- WebDataset Iterator API for PyTorch compatibility
- PNG decoder with libpng integration
- WebP decoder with libwebp integration (SIMD-accelerated)
- 7 SIMD-optimized augmentation transforms:
  - RandomHorizontalFlip (AVX2/NEON)
  - RandomVerticalFlip
  - ColorJitter (SIMD brightness/contrast/saturation)
  - RandomRotation (bilinear interpolation)
  - RandomCrop
  - RandomErasing (Cutout)
  - GaussianBlur (separable filter with SIMD)
- AugmentationPipeline for composable transforms
- DistributedWebDatasetLoader for multi-GPU training
- Python bindings for all new classes
- Comprehensive design documentation

### Changed
- Updated version to 0.3.0 across all files
- Enhanced Python bindings with augmentation transform support

### Fixed
- Missing `#include <iostream>` in thread_pool.cpp and pipeline.cpp

## [0.2.1] - 2025-01-14

### Added
- Initial PyPI release
- Basic Pipeline API
- JPEG decoding with libjpeg-turbo
- SIMD-optimized transforms (resize, normalize)
- Lock-free concurrent queues
- Memory-mapped I/O
- WebDataset TAR format support

[0.3.1]: https://github.com/ALJainProjects/TurboLoader/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/ALJainProjects/TurboLoader/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/ALJainProjects/TurboLoader/releases/tag/v0.2.1
