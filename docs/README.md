# TurboLoader Documentation

Complete documentation for TurboLoader high-performance data loading library.

---

## Quick Links

- **[API Documentation](API.md)** - Complete C++ and Python API reference
- **[Architecture Guide](ARCHITECTURE.md)** - How TurboLoader works internally
- **[Integration Guide](INTEGRATION.md)** - PyTorch/TensorFlow integration
- **[Performance Tuning](PERFORMANCE.md)** - Optimization tips
- **[Comparison Guide](COMPARISON.md)** - When to use TurboLoader vs FFCV/DALI
- **[GPU Features](GPU.md)** - GPU decode & distributed training

---

## Getting Started

### 5-Minute Quickstart

```bash
# 1. Build
mkdir build && cd build
cmake -DPython3_EXECUTABLE=/opt/homebrew/bin/python3.13 ..
make -j

# 2. Test
./tests/turboloader_tests

# 3. Benchmark
./benchmarks/basic_benchmark 8 /tmp/dataset.tar
```

See [../GETTING_STARTED.md](../GETTING_STARTED.md) for full quickstart guide.

---

## Documentation Structure

### For Users

1. **[Getting Started](../GETTING_STARTED.md)** - Installation and first steps
2. **[API Documentation](API.md)** - How to use TurboLoader in your code
3. **[Integration Guide](INTEGRATION.md)** - PyTorch/TensorFlow examples
4. **[Performance Tuning](PERFORMANCE.md)** - Optimize for your workload

### For Developers

1. **[Architecture Guide](ARCHITECTURE.md)** - Internal design and implementation
2. **[Benchmarks](../benchmarks/README.md)** - Performance comparisons

### For Decision Makers

1. **[Comparison Guide](COMPARISON.md)** - TurboLoader vs other frameworks
2. **[Benchmarks](../benchmarks/README.md)** - Verified performance data

---

## Key Features

⚡ **2.64x faster than TensorFlow**
🚀 **27.8x faster than PyTorch (naive TAR)**
💾 **81% less memory than PyTorch**
📦 **TAR streaming** (no extraction needed)
☁️ **Cloud storage** (S3/GCS support)
🧵 **C++ threads** (no Python GIL)
🔒 **Lock-free queues** (zero contention)
🎮 **GPU JPEG decode** (8.5x faster with nvJPEG)
🌐 **Distributed training** (NCCL/Gloo, multi-GPU)

---

## Examples

### Python - Basic Usage

```python
import sys
sys.path.insert(0, 'build/python')
import turboloader

pipeline = turboloader.Pipeline(
    tar_paths=['/data/train.tar'],
    num_workers=8,
    decode_jpeg=True
)

pipeline.start()
batch = pipeline.next_batch(32)

for sample in batch:
    img = sample.get_image()  # NumPy array (H, W, C)
    print(f"Image shape: {img.shape}")

pipeline.stop()
```

### C++ - Basic Usage

```cpp
#include <turboloader/pipeline/pipeline.hpp>

using namespace turboloader;

int main() {
    Pipeline::Config config{
        .num_workers = 8,
        .decode_jpeg = true
    };

    Pipeline pipeline({"/data/train.tar"}, config);
    pipeline.start();

    auto batch = pipeline.next_batch(32);
    for (const auto& sample : batch) {
        std::cout << "Image: " << sample.width << "x"
                  << sample.height << "\n";
    }

    pipeline.stop();
    return 0;
}
```

See [API Documentation](API.md) for complete examples.

---

## Performance

### Verified Benchmarks

**Data Loading** (1000 images, 256x256):
- TurboLoader: **11,628 img/s**
- TensorFlow: 9,477 img/s (1.2x slower)
- PyTorch: 400 img/s (27.8x slower)

**Memory Usage** (8 workers):
- TurboLoader: **450 MB** (shared memory)
- PyTorch: 2,400 MB (duplicated memory)
- **Savings**: 81% less memory

See [Benchmarks](../benchmarks/README.md) for full results.

---

## Framework Comparison

| Use Case | Recommended Framework |
|----------|----------------------|
| Large TAR datasets (100K+ images) | **TurboLoader** |
| Maximum performance (with preprocessing) | FFCV |
| Small datasets (<10K images) | PyTorch DataLoader |
| TensorFlow ecosystem | tf.data |
| GPU-accelerated augmentation | NVIDIA DALI |

See [Comparison Guide](COMPARISON.md) for detailed analysis.

---

## Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/turboloader/issues)
- **Questions**: See documentation above
- **Contributing**: Pull requests welcome!

---

## License

MIT License - See [LICENSE](../LICENSE)
