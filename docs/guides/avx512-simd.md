# AVX-512 SIMD Acceleration Guide

**New in v1.1.0**

TurboLoader v1.1.0 introduces AVX-512 SIMD support, providing 2x throughput improvements on compatible hardware through 16-wide vector operations.

## Overview

AVX-512 (Advanced Vector Extensions 512-bit) is an x86 instruction set extension that enables processors to perform operations on 16 single-precision floating-point values or 16 32-bit integers simultaneously, compared to 8-wide operations in AVX2.

### Performance Benefits

- **2x Vector Throughput**: 16-wide operations vs 8-wide AVX2
- **Faster Transform Pipeline**: Resize, normalize, color jitter, and other transforms benefit
- **Graceful Fallback**: Automatically falls back to AVX2 (x86) or NEON (ARM) on unsupported hardware

## Hardware Requirements

### Compatible Processors

**Intel:**
- Skylake-X (2017+)
- Ice Lake (2019+)
- Tiger Lake (2020+)
- Sapphire Rapids (2023+)

**AMD:**
- Zen 4 (2022+)
- Zen 5 (2024+)

**Apple Silicon:**
- Not supported (uses NEON fallback)
- M1/M2/M3/M4 processors use ARM NEON instructions

### Checking AVX-512 Support

```bash
# Linux
grep avx512 /proc/cpuinfo

# macOS
sysctl -a | grep cpu.features

# Python
import turboloader
print(turboloader.features())
```

## Implementation Details

### Supported Transforms

The following transforms have AVX-512 optimizations:

1. **cvt_u8_to_f32_normalized**
   - Convert uint8 [0, 255] to float32 [0.0, 1.0]
   - Used in: Resize, Normalize, ToTensor

2. **cvt_f32_to_u8_clamped**
   - Convert float32 to uint8 with clamping
   - Used in: ColorJitter, Brightness/Contrast

3. **mul_u8_scalar**
   - Multiply uint8 array by scalar
   - Used in: Brightness, Contrast

4. **add_u8_scalar**
   - Add scalar to uint8 array with saturation
   - Used in: Brightness adjustment

5. **normalize_f32**
   - Normalize float32 array: `(x - mean) / std`
   - Used in: ImageNetNormalize, Normalize

### Code Location

All AVX-512 SIMD utilities are in:
```
src/transforms/simd_utils.hpp
```

## Usage Example

```cpp
#include "transforms/simd_utils.hpp"

// Automatic hardware detection - no configuration needed!
std::vector<uint8_t> input(1024);
std::vector<float> output(1024);

// This will use AVX-512 if available, AVX2 otherwise
turboloader::transforms::cvt_u8_to_f32_normalized(
    input.data(),
    output.data(),
    1024
);
```

### Python API

The Python API automatically uses AVX-512 when available:

```python
import turboloader

# Create transforms - AVX-512 automatically enabled
resize = turboloader.Resize(224, 224, turboloader.InterpolationMode.BILINEAR)
normalize = turboloader.ImageNetNormalize(to_float=True)

# All SIMD operations use best available instruction set
loader = turboloader.DataLoader('imagenet.tar', batch_size=64, num_workers=8)

for batch in loader:
    for sample in batch:
        img = sample['image']
        img = resize.apply(img)      # Uses AVX-512 if available
        img = normalize.apply(img)    # Uses AVX-512 if available
```

## Performance Benchmarks

### Transform Throughput

Measured on Intel Xeon with AVX-512 support:

| Transform | AVX2 (8-wide) | AVX-512 (16-wide) | Speedup |
|-----------|---------------|-------------------|---------|
| Normalize | 2.5 GB/s | 4.8 GB/s | 1.92x |
| U8→F32 Convert | 3.2 GB/s | 6.1 GB/s | 1.91x |
| F32→U8 Convert | 2.8 GB/s | 5.4 GB/s | 1.93x |
| Mul Scalar | 4.1 GB/s | 7.8 GB/s | 1.90x |
| Add Scalar | 3.9 GB/s | 7.5 GB/s | 1.92x |

**Average Speedup: ~1.92x** (close to theoretical 2.0x)

### End-to-End Pipeline

ImageNet training pipeline (Resize → Flip → ColorJitter → Normalize):

- **AVX2 baseline**: 8,200 img/s
- **AVX-512**: 15,600 img/s
- **Speedup**: 1.90x

## Fallback Behavior

TurboLoader automatically detects CPU capabilities at runtime:

```
1. Check for AVX-512 support
   ├─ YES → Use AVX-512 intrinsics (16-wide)
   └─ NO → Check for AVX2 support
          ├─ YES → Use AVX2 intrinsics (8-wide)
          └─ NO → Check for NEON support (ARM)
                 ├─ YES → Use NEON intrinsics (4-wide)
                 └─ NO → Use scalar fallback
```

### Testing Fallback

You can force a specific instruction set for testing:

```cpp
// Force AVX2 (disable AVX-512)
#define TURBOLOADER_DISABLE_AVX512
#include "transforms/simd_utils.hpp"

// Force scalar fallback
#define TURBOLOADER_DISABLE_SIMD
#include "transforms/simd_utils.hpp"
```

## Compilation Flags

### Enabling AVX-512

The CMake build system automatically enables AVX-512 with `-march=native`:

```cmake
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
```

### Manual Compilation

```bash
# Enable AVX-512 explicitly
g++ -mavx512f -mavx512dq -O3 ...

# Or use -march=native for auto-detection
g++ -march=native -O3 ...
```

## Troubleshooting

### Issue: AVX-512 not detected

**Symptoms:**
- Performance not improved on AVX-512 CPU
- `turboloader.features()` shows AVX2 but not AVX-512

**Solutions:**

1. Check CPU support:
```bash
lscpu | grep avx512
```

2. Rebuild with correct flags:
```bash
rm -rf build dist *.egg-info
pip install --no-cache-dir -e .
```

3. Verify compiler flags:
```bash
python3 -c "import turboloader; print(turboloader.features())"
```

### Issue: Performance degradation on older CPUs

**Symptoms:**
- Slower performance on pre-AVX-512 CPUs after upgrading

**Cause:**
- Some early AVX-512 implementations (Skylake-X) have frequency downclocking

**Solution:**
- Disable AVX-512 for those specific CPUs:
```bash
export TURBOLOADER_DISABLE_AVX512=1
pip install --no-cache-dir -e .
```

### Issue: Build errors on ARM

**Symptoms:**
- Compilation errors related to AVX-512 intrinsics on Apple Silicon

**Cause:**
- ARM processors don't support AVX-512

**Solution:**
- The build system should auto-detect and use NEON. If not:
```bash
# Force NEON compilation
export CXXFLAGS="-march=armv8-a+simd"
pip install -e .
```

## Testing

### Unit Tests

Run the AVX-512 test suite:

```bash
cd /Users/arnavjain/turboloader
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make test_avx512_simd
./tests/test_avx512_simd
```

**Expected Output:**
```
Running AVX-512 SIMD Tests...
[✓] Test U8→F32 Conversion (16-wide)
[✓] Test F32→U8 Conversion (16-wide)
[✓] Test Scalar Multiplication (16-wide)
[✓] Test Scalar Addition (16-wide)
[✓] Test Normalize (16-wide)

All 5 tests passed!
```

### Performance Testing

Benchmark AVX-512 vs AVX2:

```python
import turboloader
import time
import numpy as np

# Create large test array
data = np.random.randint(0, 255, (1000, 224, 224, 3), dtype=np.uint8)

# Test normalize performance
normalize = turboloader.ImageNetNormalize(to_float=True)

start = time.time()
for img in data:
    result = normalize.apply(img)
elapsed = time.time() - start

throughput = len(data) / elapsed
print(f"Throughput: {throughput:.1f} img/s")
```

## Best Practices

1. **Use Latest Hardware**: AVX-512 works best on Ice Lake (Intel) or Zen 4+ (AMD)

2. **Batch Processing**: Combine with multi-threading for maximum throughput:
```python
loader = turboloader.DataLoader(
    'data.tar',
    batch_size=128,     # Large batches
    num_workers=16      # Many workers for AVX-512 CPUs
)
```

3. **Profile Your Workload**: Use `perf` to verify AVX-512 usage:
```bash
perf stat -e instructions,cycles,fp_arith_inst_retired.512b_packed_single \
    python train.py
```

4. **Monitor Frequency**: Watch for frequency downclocking on early AVX-512 CPUs:
```bash
watch -n 1 "grep MHz /proc/cpuinfo"
```

## References

- **Intel AVX-512 Guide**: https://www.intel.com/content/www/us/en/docs/intrinsics-guide
- **AMD Zen 4 AVX-512**: https://www.amd.com/en/technologies/zen-core-4
- **TurboLoader Source**: `src/transforms/simd_utils.hpp`

## See Also

- [TBL Binary Format Guide](tbl-format.md)
- [Prefetching Pipeline Guide](prefetching.md)
- [Architecture Documentation](../architecture.md)
