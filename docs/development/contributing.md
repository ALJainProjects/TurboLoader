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
