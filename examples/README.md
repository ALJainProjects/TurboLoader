# TurboLoader Examples

This directory contains practical examples demonstrating TurboLoader v1.1.0 features.

## v1.1.0 Examples

### 1. AVX-512 SIMD Performance (`avx512_performance.py`)

Demonstrates the performance benefits of AVX-512 SIMD acceleration.

**Features:**
- AVX-512 support (16-wide vector operations)
- 2x throughput improvement on compatible hardware
- Graceful fallback to AVX2/NEON
- Benchmarks for individual transforms
- Full pipeline performance testing

**Usage:**
```bash
python examples/avx512_performance.py
```

**Requirements:**
- Dataset in TAR format at `/tmp/benchmark_1000.tar`
- AVX-512 compatible CPU (optional, falls back to AVX2/NEON)

---

### 2. TBL Binary Format Conversion (`tbl_conversion.py`)

Demonstrates the TBL (TurboLoader Binary) format conversion workflow.

**Features:**
- TAR to TBL conversion
- 12.4% size reduction
- O(1) random access
- Performance comparison
- Memory-mapped I/O

**Usage:**
```bash
python examples/tbl_conversion.py /path/to/dataset.tar
```

**Output:**
- Creates `dataset.tbl` in the same directory
- Shows conversion statistics
- Demonstrates random access performance

---

### 3. Complete v1.1.0 Workflow (`complete_v110_workflow.py`)

End-to-end demonstration of all v1.1.0 features working together.

**Features:**
- TBL format conversion
- SIMD-accelerated transforms
- Prefetching pipeline
- PyTorch integration

**Usage:**
```bash
python examples/complete_v110_workflow.py /path/to/dataset.tar
```

**Steps:**
1. Convert TAR → TBL format
2. Create SIMD-accelerated pipeline
3. Benchmark complete v1.1.0 pipeline
4. Demonstrate PyTorch integration

---

## Creating a Test Dataset

If you don't have a dataset, create one with:

```bash
# Create directory and download some images
mkdir test_images
cp /path/to/your/images/*.jpg test_images/

# Create TAR file (WebDataset format)
tar cf /tmp/test_dataset.tar test_images/*.jpg

# Or use Python to generate synthetic images
python3 -c "
from PIL import Image
import tarfile
import random
from pathlib import Path

Path('test_images').mkdir(exist_ok=True)

# Generate 100 test images
for i in range(100):
    img = Image.new('RGB', (256, 256))
    pixels = img.load()
    for x in range(256):
        for y in range(256):
            pixels[x, y] = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255)
            )
    img.save(f'test_images/{i:04d}.jpg', quality=90)

# Create TAR
with tarfile.open('/tmp/test_dataset.tar', 'w') as tar:
    for i in range(100):
        tar.add(f'test_images/{i:04d}.jpg', arcname=f'{i:04d}.jpg')

print('Created /tmp/test_dataset.tar')
"
```

---

## v1.1.0 Features Summary

### 1. AVX-512 SIMD Support
- **16-wide vector operations** (2x throughput vs AVX2)
- Compatible with Intel Skylake-X+, AMD Zen 4+
- Graceful fallback to AVX2/NEON on unsupported hardware
- All transforms SIMD-optimized

### 2. TBL Binary Format
- **12.4% size reduction** vs TAR format
- **O(1) random access** via index table
- **100,000 samples/second** conversion rate
- Memory-mapped I/O for zero-copy reads
- Multi-format support (JPEG, PNG, WebP, BMP, TIFF)

### 3. Prefetching Pipeline
- Double-buffering strategy for overlapped I/O
- Thread-safe with condition variables
- Reduces epoch time by eliminating wait states
- Automatically integrated in DataLoader

---

## Additional Resources

- **Documentation:** [docs/](../docs/)
- **API Reference:** [docs/api/](../docs/api/)
- **Guides:** [docs/guides/](../docs/guides/)
  - [AVX-512 SIMD Guide](../docs/guides/avx512-simd.md)
  - [TBL Format Guide](../docs/guides/tbl-format.md)
- **Architecture:** [docs/architecture.md](../docs/architecture.md)
- **Benchmarks:** [docs/benchmarks/](../docs/benchmarks/)

---

## Support

- **Issues:** [GitHub Issues](https://github.com/ALJainProjects/TurboLoader/issues)
- **Discussions:** [GitHub Discussions](https://github.com/ALJainProjects/TurboLoader/discussions)
- **PyPI:** [https://pypi.org/project/turboloader/](https://pypi.org/project/turboloader/)

---

**TurboLoader v1.1.0** - High-performance ML data loading with SIMD acceleration
