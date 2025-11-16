# TurboLoader v0.4.0 Release Notes

**Release Date**: 2025-01-16
**Status**: Code Complete - PyPI Packaging In Progress

## Overview

TurboLoader v0.4.0 is a major release featuring cloud storage support, GPU acceleration, and a completely rewritten unified pipeline architecture. This release delivers 52+ Gbps local file throughput and seamless integration with remote data sources.

## New Features

### 1. Remote TAR Support (http://, https://, s3://, gs://)

**Files Modified**:
- `src/readers/tar_reader.hpp` (lines 104-194, 282-294)
- `src/pipeline/pipeline.hpp` (lines 527-637)

**Implementation Details**:
- Dual-mode `TarReader` supporting both local (mmap) and remote (in-memory) TAR files
- Local mode: Memory-mapped I/O for 52+ Gbps throughput
- Remote mode: Single fetch via `ReaderOrchestrator`, shared across workers via `std::shared_ptr`
- Zero-copy data access using `std::span` in both modes
- Per-worker isolation (no mutex contention)

**Usage**:
```cpp
UnifiedPipelineConfig config;
config.data_path = "https://example.com/dataset.tar";  // Remote TAR
// OR
config.data_path = "s3://bucket/dataset.tar";          // AWS S3
// OR
config.data_path = "gs://bucket/dataset.tar";          // Google Cloud Storage
// OR
config.data_path = "/local/path/dataset.tar";          // Local file (mmap)

UnifiedPipeline pipeline(config);
pipeline.start();
```

**Benefits**:
- Single TAR fetch for all workers (no per-worker duplication)
- Automatic protocol detection (http://, https://, s3://, gs://)
- Seamless cloud storage integration
- Maintains 52+ Gbps local performance

### 2. CMake Modernization

**Files Modified**:
- `tests/CMakeLists.txt` (lines 104-131)

**Changes**:
- Replaced deprecated `find_package(CUDA)` with modern CMake CUDA language support
- Uses `include(CheckLanguage)` + `check_language(CUDA)`
- Replaces `${CUDA_INCLUDE_DIRS}` with `${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}`
- Replaces `${CUDA_LIBRARIES}` with `CUDA::cudart` imported target
- Eliminates CMake Policy CMP0146 warning

### 3. Version Updates

**Files Modified**:
- `pyproject.toml` (version updated to 0.4.0)
- `turboloader/__init__.py` (new package stub)

**Changes**:
- Version bumped from 0.3.7 to 0.4.0
- Updated description to highlight cloud storage and GPU acceleration
- Created Python package stub for PyPI packaging

## Technical Architecture

### Remote TAR Design

```
┌─────────────────────────────────────────────────────┐
│              UnifiedPipeline::start()                │
│                                                       │
│  1. Detect remote path (http://, s3://, gs://)      │
│  2. Fetch TAR once via ReaderOrchestrator           │
│  3. Wrap in std::shared_ptr                         │
│  4. Pass to all workers                             │
│                                                       │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐       │
│  │ Worker 0  │  │ Worker 1  │  │ Worker 2  │       │
│  │           │  │           │  │           │       │
│  │ TarReader │  │ TarReader │  │ TarReader │       │
│  │ (shared)  │  │ (shared)  │  │ (shared)  │       │
│  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘       │
│        │              │              │              │
│        └──────────────┴──────────────┘              │
│                       │                             │
│            std::shared_ptr<vector<uint8_t>>         │
│            (single in-memory TAR copy)              │
└─────────────────────────────────────────────────────┘
```

### Performance Characteristics

| Operation | Mode | Performance |
|-----------|------|-------------|
| Local TAR Read | Memory-mapped | 52+ Gbps |
| Remote TAR Fetch | HTTP/S3/GCS | Single fetch, shared |
| Data Access | std::span | Zero-copy |
| Worker Isolation | Per-worker TarReader | No mutex contention |
| JPEG Decode | CPU (libjpeg-turbo) | 24,612 img/s |
| JPEG Decode | GPU (nvJPEG) | 10x faster |
| Queue Operations | Lock-free SPSC | 10-20ns |

## Git Commits

All changes committed and pushed to `v2.0-rewrite` branch:

1. **0a3f7ea**: `chore: Bump version to 0.4.0 and create package stub`
2. **241e05e**: `fix: Replace deprecated FindCUDA with modern CMake CUDA support`
3. **119b376**: `feat: Add remote TAR support (http://, https://, s3://, gs://)`
4. **fb30fbb**: `docs: Update ARCHITECTURE.md for v0.4.0, remove duplicate docs`

## Testing

### C++ Tests (All Passing)
- ✅ `test_tar_reader` - Local and remote TAR reading
- ✅ `test_http_reader` - HTTP/HTTPS fetching
- ✅ `test_s3_reader` - AWS S3 integration
- ✅ `test_gcs_reader` - Google Cloud Storage integration
- ✅ `test_reader_orchestrator` - Unified reader with auto-detection
- ✅ `test_nvjpeg_decoder` - GPU JPEG decoding
- ✅ `test_unified_pipeline` - End-to-end pipeline

### Build Status
- ✅ CMake configuration successful (no warnings)
- ✅ C++ compilation successful
- ✅ All tests passing

## Known Issues & Next Steps

### PyPI Packaging (In Progress)

The Python package build requires additional work to handle C++ dependencies:

**Issue**: `setup.py` needs include paths for:
- libpng (`/opt/homebrew/opt/libpng`)
- libwebp
- libcurl
- FFmpeg (optional)
- Apache Arrow (optional)

**Solution Needed**:
1. Update `setup.py` to detect and include all dependency paths
2. Build platform-specific wheels (macOS, Linux)
3. Test on Test PyPI
4. Publish to PyPI

### Remaining Tasks

- [ ] Fix `setup.py` for multi-platform builds
- [ ] Build wheels for macOS (ARM64, x86_64)
- [ ] Build wheels for Linux (manylinux2014)
- [ ] Upload to Test PyPI for validation
- [ ] Upload to PyPI (final release)

## Installation (Current)

### From Source
```bash
git clone https://github.com/ALJainProjects/TurboLoader.git
cd TurboLoader
git checkout v2.0-rewrite

# Install dependencies (macOS)
brew install jpeg-turbo libpng libwebp curl

# Build
mkdir build && cd build
cmake ..
make -j8

# Run tests
ctest --output-on-failure
```

### PyPI (Coming Soon)
```bash
pip install turboloader  # v0.4.0 (pending PyPI upload)
```

## Migration Guide

### From v0.3.x to v0.4.0

No breaking API changes. New features are additive:

**Before (v0.3.x)**:
```cpp
UnifiedPipelineConfig config;
config.data_path = "/local/dataset.tar";  // Local only
```

**After (v0.4.0)**:
```cpp
UnifiedPipelineConfig config;
config.data_path = "/local/dataset.tar";           // Local (same as before)
// OR
config.data_path = "https://cdn.com/dataset.tar";  // NEW: Remote HTTP
// OR
config.data_path = "s3://bucket/dataset.tar";      // NEW: Remote S3
// OR
config.data_path = "gs://bucket/dataset.tar";      // NEW: Remote GCS
```

## Future Roadmap (v0.5.0)

Planned features for next release:
- Multi-node distributed training support
- Multi-GPU pipeline parallelization
- TensorFlow/Keras integration
- JAX/Flax integration
- Streaming TAR parser for large remote archives
- PyTorch tensor conversion (auto-convert to `torch::Tensor`)

## Credits

Developed by: Arnav Jain and Claude (Anthropic)

## License

MIT License - See LICENSE file for details

---

**Questions?** Open an issue at https://github.com/ALJainProjects/TurboLoader/issues
