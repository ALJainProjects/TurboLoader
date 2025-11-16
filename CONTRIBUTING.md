# Contributing to TurboLoader

Thank you for your interest in contributing to TurboLoader! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/turboloader.git`
3. Create a branch: `git checkout -b feature/your-feature-name`

## Development Setup

### Prerequisites

- C++17 compatible compiler (GCC 8+, Clang 10+, MSVC 2019+)
- CMake 3.15 or higher
- Python 3.8 or higher
- libjpeg-turbo
- libpng
- libwebp
- libcurl

### Building from Source

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

### Running Tests

```bash
# C++ tests
cd build
ctest --output-on-failure

# Python tests
python -m pytest tests/
```

## Code Style

### C++
- Follow Google C++ Style Guide
- Use clang-format for formatting
- Maximum line length: 100 characters
- Use meaningful variable and function names

### Python
- Follow PEP 8
- Use type hints where applicable
- Maximum line length: 100 characters
- Use descriptive docstrings

## Pull Request Process

1. Ensure all tests pass
2. Add tests for new functionality
3. Update documentation as needed
4. Update CHANGELOG.md following Keep a Changelog format
5. Ensure your code follows the project's style guidelines
6. Write a clear PR description explaining:
   - What changes were made
   - Why the changes were necessary
   - How to test the changes

## Reporting Issues

When reporting issues, please include:

- TurboLoader version
- Operating system and version
- Python version
- Complete error message and stack trace
- Minimal reproducible example

## Feature Requests

Feature requests are welcome! Please:

- Check if the feature has already been requested
- Provide a clear use case
- Explain why existing functionality doesn't meet your needs
- Be open to discussion about implementation approaches

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Give credit where credit is due

## License

By contributing to TurboLoader, you agree that your contributions will be licensed under the MIT License.

## Questions?

Feel free to open an issue for questions about contributing!
