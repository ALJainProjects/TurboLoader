# Changelog

All notable changes to TurboLoader will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
