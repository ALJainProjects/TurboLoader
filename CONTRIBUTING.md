# Contributing to TurboLoader

Thank you for your interest in contributing to TurboLoader! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Performance Benchmarks](#performance-benchmarks)
- [Submitting Pull Requests](#submitting-pull-requests)

---

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inspiring community for all. Please be respectful and constructive in all interactions.

### Our Standards

**Examples of encouraged behavior:**
- Using welcoming and inclusive language
- Being respectful of differing viewpoints
- Gracefully accepting constructive criticism
- Focusing on what is best for the community

**Examples of unacceptable behavior:**
- Harassment or discriminatory language
- Trolling, insulting comments, or personal attacks
- Publishing others' private information without permission

---

## Getting Started

### Prerequisites

- **C++ compiler** with C++17 support (GCC 7+, Clang 5+, or MSVC 2017+)
- **CMake** 3.15 or higher
- **Python** 3.8 or higher
- **Git** for version control

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/turboloader.git
   cd turboloader
   ```
3. Add the upstream repository:
   ```bash
   git remote add upstream https://github.com/ORIGINAL_OWNER/turboloader.git
   ```

---

## Development Setup

### 1. Install Dependencies

**On macOS:**
```bash
brew install cmake libjpeg-turbo
```

**On Ubuntu/Debian:**
```bash
sudo apt-get install cmake libjpeg-turbo8-dev
```

### 2. Build from Source

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### 3. Install in Development Mode

```bash
pip install -e .
```

This allows you to edit the code and see changes without reinstalling.

### 4. Run Tests

```bash
cd build
ctest --output-on-failure
```

---

## How to Contribute

### Types of Contributions

We welcome many types of contributions:

1. **Bug reports** - Found a bug? Open an issue!
2. **Bug fixes** - Submit a PR to fix an issue
3. **New features** - Propose and implement new functionality
4. **Performance improvements** - Optimize existing code
5. **Documentation** - Improve docs, add examples
6. **Benchmarks** - Add new benchmarks or improve existing ones

### Finding an Issue to Work On

- Check the [Issues page](https://github.com/YOURUSER/turboloader/issues)
- Look for issues labeled `good first issue` or `help wanted`
- Comment on the issue to let others know you're working on it

### Creating an Issue

Before creating an issue, please:

1. **Search existing issues** to avoid duplicates
2. **Provide details:**
   - For bugs: Steps to reproduce, expected vs actual behavior, environment info
   - For features: Clear description of the use case and proposed implementation
3. **Use issue templates** when available

---

## Coding Standards

### C++ Code Style

We follow a consistent C++ coding style:

**File organization:**
- Headers in `include/turboloader/`
- Implementation in `src/`
- Tests in `tests/`

**Naming conventions:**
```cpp
class ClassName;              // PascalCase for classes
void function_name();         // snake_case for functions
int variable_name;            // snake_case for variables
constexpr int CONSTANT_NAME;  // UPPER_SNAKE_CASE for constants
```

**Formatting:**
- Indent with 4 spaces (no tabs)
- Opening braces on same line for functions/classes
- Use `auto` when type is obvious
- Prefer modern C++ (C++17 features)

**Example:**
```cpp
class DataLoader {
public:
    DataLoader(const std::string& path);

    std::optional<Sample> next_sample();

private:
    std::vector<Sample> samples_;
    size_t current_index_{0};
};
```

### Python Code Style

Follow PEP 8:

```python
class Pipeline:
    """Pipeline for data loading."""

    def __init__(self, tar_paths: List[str]):
        self.tar_paths = tar_paths

    def next_batch(self, batch_size: int) -> List[Sample]:
        """Fetch next batch of samples."""
        pass
```

**Tools:**
- Use `black` for formatting: `black .`
- Use `flake8` for linting: `flake8 .`
- Use type hints where appropriate

---

## Testing Guidelines

### Writing Tests

**C++ Tests (Google Test):**

```cpp
TEST(TarReaderTest, OpensValidFile) {
    TarReader reader("test_data/sample.tar");
    EXPECT_TRUE(reader.is_open());
}

TEST(TarReaderTest, ParsesHeaders) {
    TarReader reader("test_data/sample.tar");
    EXPECT_EQ(reader.num_samples(), 100);
}
```

**Python Tests (pytest):**

```python
def test_pipeline_creation():
    pipeline = turboloader.Pipeline(["test.tar"])
    assert pipeline is not None

def test_batch_fetching():
    pipeline = turboloader.Pipeline(["test.tar"])
    pipeline.start()
    batch = pipeline.next_batch(32)
    assert len(batch) == 32
```

### Running Tests

```bash
# C++ tests
cd build
ctest --output-on-failure

# Python tests
pytest tests/
```

### Test Coverage

Aim for:
- **Unit tests**: Test individual components in isolation
- **Integration tests**: Test components working together
- **Performance tests**: Ensure no regressions

---

## Performance Benchmarks

### Before Submitting Performance Changes

Always benchmark before and after:

```bash
# Baseline
python benchmarks/full_imagenet_benchmark.py --tar-paths test.tar --output baseline.json

# After changes (rebuild first)
make -j8
python benchmarks/full_imagenet_benchmark.py --tar-paths test.tar --output after.json

# Compare
python benchmarks/compare_results.py baseline.json after.json
```

### Performance Requirements

For performance-related PRs:
- Include benchmark results in PR description
- Show no regressions (or justify if necessary)
- Ideally show improvements of >5% to be meaningful

### Profiling

Use the detailed profiler:

```bash
python benchmarks/detailed_profiling.py test.tar --workers 16 --output profile.json
```

---

## Submitting Pull Requests

### PR Checklist

Before submitting, ensure:

- [ ] Code follows style guidelines
- [ ] All tests pass (`ctest` and `pytest`)
- [ ] New code has tests
- [ ] Documentation is updated (if applicable)
- [ ] Benchmark results included (for performance changes)
- [ ] Commit messages are clear and descriptive
- [ ] PR description explains the changes

### PR Process

1. **Create a branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes:**
   - Write code
   - Add tests
   - Update documentation

3. **Commit your changes:**
   ```bash
   git add .
   git commit -m "Add feature: brief description

   Detailed explanation of what changed and why."
   ```

4. **Push to your fork:**
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Create Pull Request:**
   - Go to GitHub and create PR from your branch
   - Fill out the PR template
   - Link any related issues

6. **Code Review:**
   - Address reviewer comments
   - Update your PR as needed
   - Be patient and responsive

### Commit Message Format

```
<type>: <short summary> (50 chars or less)

<detailed description of what changed and why>

<optional footer with issue references>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `perf`: Performance improvement
- `docs`: Documentation changes
- `test`: Adding/updating tests
- `refactor`: Code refactoring
- `ci`: CI/CD changes

**Example:**
```
feat: Add NEON SIMD support for ARM processors

Implemented ARM NEON intrinsics for image resize and normalization
operations. This provides 4-6x speedup on Apple M1/M2 processors.

- Added neon-specific implementations in simd_transforms.cpp
- Added CPU feature detection for ARM
- Added benchmarks for ARM platforms

Closes #42
```

---

## Specific Contribution Areas

### 1. SIMD Optimizations

When adding new SIMD code:

- Support both x86 (AVX2/AVX-512) and ARM (NEON)
- Provide scalar fallback for unsupported platforms
- Benchmark on multiple architectures
- Document expected speedups

**Example PR:** "Add AVX-512 support for normalization"

### 2. New Transforms

When adding transforms (crop, flip, color conversion):

- Follow existing transform API
- Support both uint8 and float input
- Include SIMD implementation if beneficial
- Add comprehensive tests

**Example PR:** "Add random crop augmentation"

### 3. Documentation

Documentation improvements are always welcome:

- Fix typos or unclear explanations
- Add examples and tutorials
- Improve API documentation
- Create diagrams or visualizations

### 4. Benchmarks

Add benchmarks for new datasets or scenarios:

- Compare with PyTorch DataLoader
- Include multiple worker counts
- Test on different hardware
- Generate nice visualizations

---

## Platform-Specific Contributions

### macOS / Apple Silicon

- Test on M1/M2/M3 processors
- Verify NEON optimizations work
- Check ARM-specific code paths

### Linux

- Test on various distributions (Ubuntu, CentOS, Arch)
- Verify with different C++ compilers (GCC, Clang)
- Test with different SIMD instruction sets

### Windows (experimental)

Windows support is experimental. Contributions welcome:

- MSVC compatibility fixes
- Windows-specific build issues
- Path handling differences

---

## Getting Help

If you need help:

1. **Read the documentation** - Check README, ARCHITECTURE.md
2. **Search existing issues** - Someone might have asked already
3. **Ask in Discussions** - Use GitHub Discussions for questions
4. **Join community chat** - [Link to Discord/Slack if available]

---

## Recognition

Contributors are recognized in:

- `AUTHORS.md` file
- Release notes
- Social media announcements (with permission)

Thank you for contributing to TurboLoader!

---

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (MIT License).
