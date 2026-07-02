# Contributing to TurboLoader

Thank you for your interest in contributing to TurboLoader!

## Development Setup

### 1. Clone Repository

```bash
git clone https://github.com/ALJainProjects/TurboLoader.git
cd TurboLoader
```

### 2. Install Dependencies

**macOS:**
```bash
brew install cmake libjpeg-turbo libpng libwebp
```

**Ubuntu:**
```bash
sudo apt-get install cmake libjpeg-turbo8-dev libpng-dev libwebp-dev
```

### 3. Build from Source

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j8
```

### 4. Install Python Package

```bash
cd ..
pip install -e .
```

> **End users don't need to build from source.** `pip install turboloader` pulls prebuilt
> manylinux wheels for Linux `x86_64` and `aarch64` (plus an sdist); portable macOS wheels
> built from source are being added. PyTorch is an optional dependency — install it with
> `pip install turboloader[torch]` only if you need the PyTorch tensor output paths.
> Building from source (above) is for contributors and for enabling optional backends
> (remote/cloud readers, GPU, HDF5, Zarr, TFRecord) that are off in the prebuilt wheels.

## Running Tests

### C++ Tests

```bash
cd build
./tests/test_transforms
./tests/test_advanced_transforms
```

### Python Tests

```bash
pytest tests/test_pytorch_transforms.py -v
pytest tests/test_transforms_tensorflow.py -v
```

## Versioning & Releases

- The version is managed by `setuptools_scm` and derived from the git tag — there is **no
  hardcoded version string** to bump. `turboloader.version()`, `__version__`, and
  `features()['version']` all resolve to the same value.
- Releases are published to PyPI via GitHub Trusted Publishing, triggered by pushing a
  version tag. Tagging a release is what cuts the wheels and uploads them.
- The current version is 2.31.0 (the tag is the single source of truth via setuptools_scm).

## Code Style

- **C++:** Follow Google C++ Style Guide
- **Python:** Follow PEP 8
- **Documentation:** Clear, concise, with examples

## Pull Request Process

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## Questions?

- [GitHub Issues](https://github.com/ALJainProjects/TurboLoader/issues)
- [GitHub Discussions](https://github.com/ALJainProjects/TurboLoader/discussions)
