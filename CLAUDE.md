# TurboLoader

High-performance ML data loading library. C++20 core with Python bindings via pybind11. Targets 12x faster throughput than PyTorch DataLoader.

## Build Commands

```bash
# C++ build
mkdir -p build && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j$(nproc)

# Python install (editable)
pip install -e .

# C++ tests
cd build && ctest --output-on-failure -j$(nproc)

# Python tests
pytest tests/ -v --tb=short

# Lint
black --check turboloader/ tests/
flake8 turboloader/ tests/ --select=E9,F63,F7,F82
```

## Architecture

### Source (`src/`)
- `core/` - Lock-free SPSC queues, buffer pools, hybrid wait strategies
- `pipeline/` - Main data pipeline, prefetching, smart batching, GPU prefetch, checkpointing
- `decode/` - Image/audio/video/CSV/text decoders (JPEG via libjpeg-turbo, nvJPEG for GPU)
- `transforms/` - 24 SIMD-accelerated transforms (AVX2/NEON), AutoAugment, TrivialAugment
- `readers/` - TAR, TBL v2, HTTP, S3, GCS, Azure, LMDB readers
- `writers/` - TBL v2 streaming writer
- `formats/` - Format handling (HDF5, TFRecord, Zarr, WebDataset)
- `distributed/` - Sharding strategies for DDP/Horovod/DeepSpeed
- `sampling/` - Weighted sampling, quasi-random sampling
- `gpu/` - GPU decode pipeline, multi-GPU support
- `io/` - io_uring async I/O
- `python/` - pybind11 bindings (`turboloader_bindings.cpp`)

### Python (`turboloader/`)
- `__init__.py` - Python wrappers (DataLoader, FastDataLoader, MemoryEfficientDataLoader, ProgressiveResizeLoader)
- `pytorch_compat.py` - PyTorch drop-in compatibility (PyTorchCompatibleLoader, ImageFolderConverter, TransformAdapter)

### Tests (`tests/`)
- C++ tests: `test_*.cpp` (built via CMake/GoogleTest)
- Python tests: `test_*.py` (pytest)

## Code Style

### C++
- C++20, Google C++ Style Guide
- 100 char line limit
- Header-only in `src/` (`.hpp` files)
- Always include required standard headers explicitly (e.g., `<cstring>` for `std::memcpy`)
- Use `#pragma once` for header guards
- Wrap everything in `namespace turboloader {}`
- Use `#ifdef TURBOLOADER_HAS_CUDA` for GPU-conditional code

### Python
- PEP 8, black formatting (100 char line limit)
- Type hints where applicable
- The C++ extension is `_turboloader`, Python wraps it in `turboloader/`

## Version Management

Version must be updated in three places:
1. `pyproject.toml` - `version = "X.Y.Z"`
2. `setup.py` - `version="X.Y.Z"`
3. `turboloader/__init__.py` - `__version__ = "X.Y.Z"`

Tags trigger CI: `git tag -a vX.Y.Z -m "message" && git push origin vX.Y.Z`

## CI/CD

- `test.yml` - Runs on push/PR to main: Python tests (3.10-3.12), C++ tests, lint
- `build-wheels.yml` - Triggered by `v*` tags: builds wheels via cibuildwheel
- `claude-review.yml` - Claude Code review on PRs and @claude mentions

## Dependencies

System: libjpeg-turbo, libpng, libwebp, libcurl, liblz4, cmake
Python: numpy, torch, pybind11

## Common Pitfalls

- Missing C++ standard headers cause GCC failures but may silently work on Clang/macOS (e.g., `<cstring>` for `std::memcpy`, `<cstdlib>` for `std::free`)
- Python tests use `|| true` in CI to not fail the build on partial test failures
- The `_turboloader` C++ module must be built before Python imports work
- `FastDataLoader` returns `(np.ndarray, dict)` while `DataLoader` returns `List[Dict]`
