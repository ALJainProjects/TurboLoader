# TurboLoader Troubleshooting Guide

This guide covers common issues and their solutions when using TurboLoader.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Data Loading Problems](#data-loading-problems)
- [Performance Issues](#performance-issues)
- [Transform Errors](#transform-errors)
- [Memory Issues](#memory-issues)
- [Distributed Training](#distributed-training)
- [Platform-Specific Issues](#platform-specific-issues)
- [Integration Issues](#integration-issues)

---

## Installation Issues

### ImportError: No module named 'turboloader'

**Problem**: Cannot import turboloader after installation.

**Solutions**:
1. Verify installation:
   ```bash
   pip show turboloader
   python -c "import turboloader; print(turboloader.__version__)"
   ```

2. Check you're using the correct Python environment:
   ```bash
   which python
   pip list | grep turboloader
   ```

3. Reinstall from PyPI:
   ```bash
   pip uninstall turboloader
   pip install turboloader
   ```

### Build Errors on Installation

**Problem**: Compilation errors when installing from source.

**Solutions**:
1. Ensure you have C++20 compiler:
   ```bash
   # macOS
   clang++ --version  # Should be 14.0+

   # Linux
   g++ --version  # Should be 11.0+
   ```

2. Install required dependencies:
   ```bash
   # macOS
   brew install cmake jpeg-turbo libpng

   # Ubuntu/Debian
   sudo apt-get install build-essential cmake libjpeg-turbo8-dev libpng-dev

   # RHEL/CentOS
   sudo yum install gcc-c++ cmake libjpeg-turbo-devel libpng-devel
   ```

3. Install from PyPI instead (pre-built wheels):
   ```bash
   pip install turboloader
   ```

### Wrong Version After Update

**Problem**: Still seeing old version after updating.

**Solutions**:
1. Force reinstall:
   ```bash
   pip install --force-reinstall --no-cache-dir turboloader
   ```

2. Check for multiple installations:
   ```bash
   pip list | grep turboloader
   pip3 list | grep turboloader
   python -m pip show turboloader
   ```

3. Remove all versions and reinstall:
   ```bash
   pip uninstall turboloader -y
   pip install turboloader
   ```

---

## Data Loading Problems

### "Failed to open TAR archive" Error

**Problem**: DataLoader cannot read TAR file.

**Solutions**:
1. Verify TAR file exists and is readable:
   ```bash
   ls -lh /path/to/dataset.tar
   file /path/to/dataset.tar
   ```

2. Check TAR file integrity:
   ```bash
   tar -tzf /path/to/dataset.tar | head
   ```

3. Ensure TAR is in correct format (POSIX ustar):
   ```python
   import tarfile
   with tarfile.open('/path/to/dataset.tar', 'r') as tar:
       print(f"Format: {tar.format}")
       print(f"Members: {len(tar.getmembers())}")
   ```

### Empty Batches or No Data

**Problem**: DataLoader returns empty batches or exits immediately.

**Solutions**:
1. Check TAR file has valid image files:
   ```bash
   tar -tzf dataset.tar | grep -E '\.(jpg|jpeg|png)$' | head
   ```

2. Verify batch_size and dataset size:
   ```python
   import turboloader
   loader = turboloader.DataLoader(
       'dataset.tar',
       batch_size=32,  # Try smaller batch
       num_workers=1   # Reduce workers for debugging
   )

   count = 0
   for batch in loader:
       count += len(batch)
       print(f"Batch {count}: {len(batch)} samples")
       if count >= 100:
           break
   ```

3. Check for corrupted images:
   ```python
   for batch in loader:
       for sample in batch:
           img = sample['image']
           if img is None or img.size == 0:
               print(f"Corrupted image: {sample.get('filename', 'unknown')}")
   ```

### "JPEG decode error" or Image Corruption

**Problem**: Images fail to decode or appear corrupted.

**Solutions**:
1. Validate JPEG files in TAR:
   ```bash
   # Extract and test first few images
   tar -xzf dataset.tar --to-stdout $(tar -tzf dataset.tar | head -1) | file -
   ```

2. Use error handling:
   ```python
   loader = turboloader.DataLoader(
       'dataset.tar',
       batch_size=32,
       num_workers=4
   )

   for batch in loader:
       for sample in batch:
           try:
               img = sample['image']
               if img.shape[0] == 0 or img.shape[1] == 0:
                   print(f"Invalid dimensions: {sample.get('filename')}")
           except Exception as e:
               print(f"Error processing {sample.get('filename')}: {e}")
   ```

3. Check JPEG library version:
   ```python
   import turboloader
   print(turboloader.__version__)
   # Ensure using TurboJPEG for best compatibility
   ```

---

## Performance Issues

### Slow Data Loading Speed

**Problem**: DataLoader is slower than expected.

**Solutions**:
1. Optimize worker count:
   ```python
   # Try different worker counts
   for num_workers in [1, 2, 4, 8, 16]:
       loader = turboloader.DataLoader(
           'dataset.tar',
           batch_size=32,
           num_workers=num_workers
       )

       import time
       start = time.time()
       count = 0
       for batch in loader:
           count += len(batch)
           if count >= 1000:
               break

       elapsed = time.time() - start
       throughput = count / elapsed
       print(f"Workers: {num_workers}, Throughput: {throughput:.1f} img/s")
   ```

2. Increase batch size:
   ```python
   # Larger batches = better throughput
   loader = turboloader.DataLoader(
       'dataset.tar',
       batch_size=128,  # Increase if GPU memory allows
       num_workers=8
   )
   ```

3. Use TBL format for faster loading:
   ```python
   # Convert TAR to TBL v2
   writer = turboloader.TblWriterV2('dataset.tbl', compression=True)
   reader = turboloader.DataLoader('dataset.tar', batch_size=1, num_workers=1)

   for batch in reader:
       for sample in batch:
           writer.add_sample(
               data=sample['image'],
               format=turboloader.SampleFormat.JPEG,
               metadata={'label': sample.get('label', 0)}
           )
   writer.finalize()

   # Load from TBL (faster)
   loader = turboloader.DataLoader('dataset.tbl', batch_size=128, num_workers=8)
   ```

### CPU Bottleneck During Training

**Problem**: GPU utilization is low, CPU is saturated.

**Solutions**:
1. Increase prefetch buffer:
   ```python
   loader = turboloader.DataLoader(
       'dataset.tar',
       batch_size=64,
       num_workers=8,
       prefetch_factor=4  # Increase buffer size
   )
   ```

2. Use pinned memory for faster GPU transfer:
   ```python
   import torch

   for batch in loader:
       images = torch.stack([torch.from_numpy(s['image']) for s in batch])
       images = images.pin_memory().to('cuda', non_blocking=True)
   ```

3. Apply transforms in C++ (faster than Python):
   ```python
   # Use TurboLoader's SIMD transforms instead of torchvision
   transforms = turboloader.Compose([
       turboloader.Resize(256, 256),
       turboloader.RandomCrop(224, 224),
       turboloader.RandomHorizontalFlip(0.5),
       turboloader.ImageNetNormalize()
   ])
   ```

---

## Transform Errors

### "TypeError: incompatible constructor arguments"

**Problem**: Transform initialization fails.

**Solutions**:
1. Check parameter types:
   ```python
   # Wrong - single dimension
   transform = turboloader.Resize(256)

   # Correct - both width and height
   transform = turboloader.Resize(256, 256)
   ```

2. Common transform signatures:
   ```python
   # Resize
   turboloader.Resize(width, height)

   # RandomCrop
   turboloader.RandomCrop(width, height)

   # RandomHorizontalFlip
   turboloader.RandomHorizontalFlip(probability)

   # ColorJitter
   turboloader.ColorJitter(brightness, contrast, saturation, hue)
   ```

### Transform Order Matters

**Problem**: Unexpected results from transform pipeline.

**Solutions**:
1. Apply transforms in correct order:
   ```python
   # Correct order
   transforms = turboloader.Compose([
       turboloader.Resize(256, 256),        # 1. Resize first
       turboloader.RandomCrop(224, 224),    # 2. Then crop
       turboloader.RandomHorizontalFlip(0.5), # 3. Spatial augmentations
       turboloader.ColorJitter(0.2, 0.2, 0.2, 0.1), # 4. Color augmentations
       turboloader.ImageNetNormalize()      # 5. Normalize last
   ])
   ```

2. Avoid destructive transforms early:
   ```python
   # Wrong - crop before resize loses data
   transforms = turboloader.Compose([
       turboloader.RandomCrop(100, 100),
       turboloader.Resize(256, 256)  # Upscaling cropped image
   ])
   ```

---

## Memory Issues

### Out of Memory (OOM) Errors

**Problem**: Process runs out of memory during data loading.

**Solutions**:
1. Reduce batch size:
   ```python
   loader = turboloader.DataLoader(
       'dataset.tar',
       batch_size=16,  # Reduce from 128
       num_workers=4
   )
   ```

2. Reduce worker count:
   ```python
   # Each worker uses memory
   loader = turboloader.DataLoader(
       'dataset.tar',
       batch_size=32,
       num_workers=2  # Reduce workers
   )
   ```

3. Monitor memory usage:
   ```python
   import psutil
   import os

   process = psutil.Process(os.getpid())

   for i, batch in enumerate(loader):
       mem_mb = process.memory_info().rss / 1024 / 1024
       print(f"Batch {i}: Memory usage: {mem_mb:.1f} MB")
   ```

### Memory Leak Over Time

**Problem**: Memory usage grows continuously during training.

**Solutions**:
1. Explicitly delete batches:
   ```python
   for batch in loader:
       # Process batch
       images = process_batch(batch)

       # Clean up
       del batch
       del images

       if i % 100 == 0:
           import gc
           gc.collect()
   ```

2. Recreate loader periodically:
   ```python
   for epoch in range(num_epochs):
       # Create new loader each epoch
       loader = turboloader.DataLoader(
           'dataset.tar',
           batch_size=32,
           num_workers=4,
           shuffle=True
       )

       for batch in loader:
           # Training code
           pass

       # Loader auto-cleanup at end of scope
   ```

---

## Distributed Training

### Duplicate Data Across Ranks

**Problem**: Different GPUs process the same data.

**Solutions**:
1. Enable distributed mode:
   ```python
   import torch.distributed as dist

   dist.init_process_group(backend='nccl')
   rank = dist.get_rank()
   world_size = dist.get_world_size()

   loader = turboloader.DataLoader(
       'dataset.tar',
       batch_size=32,
       num_workers=4,
       enable_distributed=True,  # Important!
       drop_last=True
   )
   ```

2. Verify sharding:
   ```python
   # Print sample IDs on each rank
   for batch in loader:
       for sample in batch:
           print(f"Rank {rank}: {sample.get('filename', 'unknown')}")
       break
   ```

### Inconsistent Epoch Lengths

**Problem**: Different ranks finish at different times.

**Solutions**:
1. Use `drop_last=True`:
   ```python
   loader = turboloader.DataLoader(
       'dataset.tar',
       batch_size=32,
       num_workers=4,
       enable_distributed=True,
       drop_last=True  # Ensure consistent batches
   )
   ```

2. Manual synchronization:
   ```python
   for batch in loader:
       # Process batch
       loss = train_step(batch)

       # Sync all ranks
       dist.barrier()
   ```

---

## Platform-Specific Issues

### macOS: "Library not loaded" Error

**Problem**: Cannot load dynamic libraries on macOS.

**Solutions**:
1. Install Homebrew dependencies:
   ```bash
   brew install jpeg-turbo libpng
   ```

2. Set library path:
   ```bash
   export DYLD_LIBRARY_PATH=/opt/homebrew/lib:$DYLD_LIBRARY_PATH
   ```

### Linux: "GLIBC version not found"

**Problem**: Incompatible GLIBC version.

**Solutions**:
1. Check GLIBC version:
   ```bash
   ldd --version
   ```

2. Install from source with compatible flags:
   ```bash
   pip install turboloader --no-binary turboloader
   ```

3. Use manylinux wheel:
   ```bash
   pip install turboloader --only-binary :all:
   ```

### Windows: Build Not Supported

**Problem**: TurboLoader doesn't officially support Windows yet.

**Workaround**:
- Use WSL2 (Windows Subsystem for Linux)
- Use Docker with Linux container
- Track Windows support issue: [GitHub Issues](https://github.com/ALJainProjects/TurboLoader/issues)

---

## Integration Issues

### PyTorch Lightning: "Expected DataLoader, got NoneType"

**Problem**: Lightning expects standard DataLoader.

**Solution**: Wrap TurboLoader as IterableDataset:
```python
from torch.utils.data import IterableDataset, DataLoader as TorchDataLoader

class TurboLoaderWrapper(IterableDataset):
    def __init__(self, data_path, batch_size, num_workers):
        self.loader = turboloader.DataLoader(
            data_path,
            batch_size=batch_size,
            num_workers=num_workers
        )

    def __iter__(self):
        for batch in self.loader:
            for sample in batch:
                yield sample['image'], sample.get('label', 0)

# Use with Lightning
dataset = TurboLoaderWrapper('dataset.tar', 32, 4)
loader = TorchDataLoader(dataset, batch_size=None, num_workers=0)
```

See full example: `examples/pytorch_lightning_example.py`

### TensorFlow: Mutex Lock Error on Shutdown

**Problem**: "mutex lock failed: Invalid argument" when script exits.

**Solution**: Add graceful cleanup:
```python
if __name__ == '__main__':
    try:
        main()
    finally:
        import gc
        gc.collect()
        import sys
        sys.exit(0)
```

### JAX: Array Conversion Issues

**Problem**: NumPy arrays from TurboLoader not compatible with JAX.

**Solution**: Explicit conversion:
```python
import jax.numpy as jnp

for batch in loader:
    images = [jnp.array(sample['image']) for sample in batch]
    images = jnp.stack(images)
```

---

## FAQ

### Q: How do I check if SIMD acceleration is enabled?

**A**: Check compile flags:
```python
import turboloader
print(turboloader.__version__)
# SIMD is enabled by default on x86_64 (AVX2) and ARM64 (NEON)
```

Run benchmarks to verify performance:
```bash
python benchmarks/run_all_benchmarks.py
```

### Q: Can I use TurboLoader without PyTorch/TensorFlow?

**A**: Yes! TurboLoader is framework-agnostic:
```python
import turboloader

loader = turboloader.DataLoader('dataset.tar', batch_size=32, num_workers=4)

for batch in loader:
    for sample in batch:
        img = sample['image']  # NumPy array
        # Process with any framework or custom code
```

### Q: How do I handle variable-size images?

**A**: TurboLoader preserves original dimensions:
```python
for batch in loader:
    for sample in batch:
        img = sample['image']
        h, w, c = img.shape
        print(f"Image size: {w}x{h}")

        # Resize if needed
        if h != 224 or w != 224:
            resize = turboloader.Resize(224, 224)
            img = resize.apply(img)
```

### Q: What's the difference between TAR and TBL format?

**A**:
- **TAR**: Standard archive format, portable, larger file size
- **TBL v2**: TurboLoader's optimized format, 40-60% smaller with LZ4 compression, faster loading

Use TAR for portability, TBL for production performance.

### Q: How do I debug which images are corrupted?

**A**:
```python
loader = turboloader.DataLoader('dataset.tar', batch_size=1, num_workers=1)

for i, batch in enumerate(loader):
    for sample in batch:
        try:
            img = sample['image']
            if img.size == 0:
                print(f"Empty image: {sample.get('filename')}")
        except Exception as e:
            print(f"Error at index {i}: {sample.get('filename')}: {e}")
```

### Q: Does TurboLoader support video/audio?

**A**: Currently in development. Subscribe to:
- [GitHub Discussions](https://github.com/ALJainProjects/TurboLoader/discussions)
- [GitHub Issues](https://github.com/ALJainProjects/TurboLoader/issues)

---

## Getting Help

If your issue isn't covered here:

1. **Search existing issues**: [GitHub Issues](https://github.com/ALJainProjects/TurboLoader/issues)
2. **Ask in discussions**: [GitHub Discussions](https://github.com/ALJainProjects/TurboLoader/discussions)
3. **File a bug report**: Use issue template at `.github/ISSUE_TEMPLATE/bug_report.yml`
4. **Check documentation**: [README.md](https://github.com/ALJainProjects/TurboLoader/blob/main/README.md)

When reporting issues, please include:
- TurboLoader version
- Python version
- Operating system
- Minimal reproducible example
- Full error message and stack trace
