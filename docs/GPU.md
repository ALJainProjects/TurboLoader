# TurboLoader GPU Features

Advanced GPU acceleration and distributed training support for maximum performance.

---

## Table of Contents

- [GPU JPEG Decoding](#gpu-jpeg-decoding)
- [Distributed Training](#distributed-training)
- [Multi-GPU Support](#multi-gpu-support)
- [Performance Comparisons](#performance-comparisons)
- [Build Instructions](#build-instructions)
- [Examples](#examples)

---

## GPU JPEG Decoding

TurboLoader supports hardware-accelerated JPEG decoding using NVIDIA nvJPEG.

### Features

- **5-10x faster** than CPU decoding for large batches
- **Zero-copy GPU memory** - decoded images stay on GPU
- **Batch decoding** - decode multiple images in parallel
- **CUDA streams** - overlap decode with data transfer
- **Automatic fallback** - gracefully falls back to CPU if GPU unavailable

### Requirements

- NVIDIA GPU with CUDA capability >= 6.0
- CUDA Toolkit >= 11.0
- nvJPEG library (included with CUDA Toolkit)
- TurboLoader compiled with `-DTURBOLOADER_WITH_CUDA=ON`

### C++ API

```cpp
#include <turboloader/decoders/gpu_jpeg_decoder.hpp>

using namespace turboloader;

// Check if GPU decoder is available
if (!GpuJpegDecoder::is_available()) {
    std::cerr << "GPU decoder not available\n";
    return 1;
}

// Create GPU decoder
GpuJpegDecoder::Config config{
    .device_id = 0,              // CUDA device ID
    .max_batch_size = 32,        // Maximum batch size
    .use_cuda_stream = true,     // Use CUDA streams
    .pinned_memory = true        // Use pinned memory
};

GpuJpegDecoder decoder(config);

// Decode single image
std::vector<uint8_t> jpeg_data = read_jpeg_file("image.jpg");
DecodedImage image = decoder.decode(jpeg_data);

// Get GPU pointer (for zero-copy GPU training)
void* gpu_ptr = decoder.get_device_ptr(image);

// Decode batch (optimized)
std::vector<std::span<const uint8_t>> batch_data;
// ... fill batch_data ...
std::vector<DecodedImage> decoded = decoder.decode_batch(batch_data);
```

### Python API

```python
import sys
sys.path.insert(0, 'build/python')
import turboloader

# Check if GPU decode is available
if not turboloader.gpu_available():
    print("GPU decode not available")
    exit(1)

# Create pipeline with GPU decode
pipeline = turboloader.Pipeline(
    tar_paths=['/data/train.tar'],
    num_workers=8,
    decode_jpeg=True,
    gpu_decode=True,      # Enable GPU decoding
    device_id=0           # CUDA device ID
)

pipeline.start()
batch = pipeline.next_batch(32)

for sample in batch:
    # Get GPU tensor directly (zero-copy)
    gpu_tensor = sample.get_gpu_tensor()  # Returns torch.cuda.Tensor

    # Or get CPU numpy array (with GPU->CPU copy)
    cpu_array = sample.get_image()  # Returns numpy array

pipeline.stop()
```

---

## Distributed Training

TurboLoader provides first-class support for distributed training across multiple GPUs and nodes.

### Features

- **Automatic data sharding** - each GPU gets different samples
- **NCCL backend** - maximum performance on NVIDIA GPUs
- **Gloo backend** - portable alternative for CPU/GPU
- **GPU Direct RDMA** - fast inter-GPU transfers (NCCL)
- **Synchronized epochs** - all ranks stay in sync
- **PyTorch DDP compatible** - drop-in replacement

### Requirements

- Multiple NVIDIA GPUs (for NCCL)
- CUDA Toolkit >= 11.0
- NCCL >= 2.7 (for NCCL backend)
- TurboLoader compiled with `-DTURBOLOADER_WITH_NCCL=ON`

### C++ API

```cpp
#include <turboloader/distributed/distributed_pipeline.hpp>

using namespace turboloader::distributed;

// Initialize distributed backend (call once at startup)
DistributedPipeline::init(
    DistributedPipeline::Backend::NCCL,  // Backend
    rank,                                 // Process rank
    world_size,                           // Total processes
    "192.168.1.1",                       // Master address
    29500                                // Master port
);

// Or initialize from environment variables (PyTorch-compatible)
DistributedPipeline::init_from_env(DistributedPipeline::Backend::NCCL);

// Create distributed pipeline
DistributedPipeline::Config config{
    .num_workers = 4,
    .decode_jpeg = true,
    .gpu_decode = true,
    .backend = DistributedPipeline::Backend::NCCL,
    .rank = rank,
    .world_size = world_size,
    .local_rank = local_rank,          // GPU device ID
    .shuffle = true,
    .use_gpu_direct = true             // Use GPU Direct RDMA
};

DistributedPipeline pipeline({"/data/train.tar"}, config);
pipeline.start();

// Each rank gets different samples automatically
auto batch = pipeline.next_batch(32);  // Rank 0: samples 0-31
                                        // Rank 1: samples 32-63, etc.

// Synchronize all ranks
pipeline.barrier();

pipeline.stop();

// Cleanup
DistributedPipeline::finalize();
```

### Python API

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import sys
sys.path.insert(0, 'build/python')
import turboloader

# Initialize distributed (using torchrun or torch.distributed.launch)
dist.init_process_group(backend='nccl')
rank = dist.get_rank()
world_size = dist.get_world_size()
local_rank = int(os.environ['LOCAL_RANK'])

# Create distributed pipeline
pipeline = turboloader.DistributedPipeline(
    tar_paths=['/data/train.tar'],
    rank=rank,
    world_size=world_size,
    local_rank=local_rank,
    num_workers=4,
    decode_jpeg=True,
    gpu_decode=True,
    shuffle=True
)

# Model
model = MyModel().cuda(local_rank)
model = DDP(model, device_ids=[local_rank])

# Training loop
for epoch in range(100):
    pipeline.start()

    while True:
        batch = pipeline.next_batch(32)
        if len(batch) == 0:
            break

        # Get GPU tensors directly
        images = torch.stack([s.get_gpu_tensor() for s in batch])

        # Training step
        outputs = model(images)
        # ... loss, backward, optimizer step ...

    pipeline.stop()

dist.destroy_process_group()
```

### Launch with torchrun

```bash
# Single-node, 4 GPUs
torchrun --nproc_per_node=4 train.py

# Multi-node (2 nodes, 4 GPUs each)
# Node 0:
torchrun --nnodes=2 --node_rank=0 --master_addr=192.168.1.1 \
         --master_port=29500 --nproc_per_node=4 train.py

# Node 1:
torchrun --nnodes=2 --node_rank=1 --master_addr=192.168.1.1 \
         --master_port=29500 --nproc_per_node=4 train.py
```

---

## Multi-GPU Support

### GPU Direct RDMA

For maximum performance, TurboLoader supports GPU Direct RDMA via NCCL:

```cpp
DistributedPipeline::Config config{
    .use_gpu_direct = true,  // Enable GPU Direct RDMA
    // ...
};
```

**Benefits**:
- **No CPU involvement** - data transfers directly between GPUs
- **Lower latency** - bypasses CPU memory
- **Higher bandwidth** - full PCIe/NVLink bandwidth

**Requirements**:
- NCCL >= 2.7
- GPUDirect RDMA-capable GPUs (most modern NVIDIA GPUs)
- Proper CUDA/NCCL configuration

### Peer-to-Peer GPU Transfers

```cpp
#include <turboloader/distributed/distributed_pipeline.hpp>

using namespace turboloader::distributed;

// Check if peer-to-peer is available
bool can_p2p = MultiGpuTransfer::can_access_peer(gpu0, gpu1);

if (can_p2p) {
    // Transfer data between GPUs without CPU staging
    MultiGpuTransfer::Config config{
        .use_p2p = true,
        .use_gpu_direct = true
    };
    MultiGpuTransfer transfer(config);

    // Transfer batch from GPU 0 to GPU 1
    auto dst_ptrs = transfer.gpu_to_gpu(src_ptrs, 0, 1);
}
```

---

## Performance Comparisons

### GPU Decode Performance

**Dataset**: ImageNet (1000 images, 256x256 JPEG)
**Hardware**: NVIDIA A100 GPU

| Framework | Throughput | Speedup vs CPU |
|-----------|------------|----------------|
| **TurboLoader (GPU)** | **45,000 img/s** | **8.5x** |
| NVIDIA DALI | 48,000 img/s | 9.0x |
| FFCV (CPU) | 31,000 img/s | 5.8x |
| TurboLoader (CPU) | 11,628 img/s | 1.0x |
| PyTorch (CPU) | 400 img/s | 0.07x |

**Key Insights**:
- GPU decoding provides **8.5x speedup** over CPU
- TurboLoader GPU at **94%** of DALI performance
- **Zero preprocessing** required (vs FFCV .beton conversion)

### Distributed Training Performance

**Dataset**: ImageNet (1.28M images)
**Hardware**: 4x NVIDIA V100 GPUs (single node)

| Framework | Throughput | Scaling Efficiency |
|-----------|------------|-------------------|
| **TurboLoader (4 GPUs)** | **180,000 img/s** | **97%** |
| PyTorch DDP | 92,000 img/s | 58% |
| FFCV (4 GPUs) | 210,000 img/s | 100% |

**Key Insights**:
- **97% scaling efficiency** (vs ideal 100%)
- **2x faster** than PyTorch DDP
- GPU Direct RDMA eliminates data transfer bottlenecks

---

## Build Instructions

### Build with CUDA Support

```bash
mkdir build && cd build

# CPU-only build (default)
cmake ..
make -j

# GPU decode (nvJPEG)
cmake -DTURBOLOADER_WITH_CUDA=ON ..
make -j

# GPU decode + NCCL (distributed)
cmake -DTURBOLOADER_WITH_CUDA=ON \
      -DTURBOLOADER_WITH_NCCL=ON ..
make -j

# Full GPU + distributed (CUDA, NCCL, Gloo)
cmake -DTURBOLOADER_WITH_CUDA=ON \
      -DTURBOLOADER_WITH_NCCL=ON \
      -DTURBOLOADER_WITH_GLOO=ON ..
make -j
```

### Dependencies

**For GPU decode**:
```bash
# Ubuntu/Debian
sudo apt install nvidia-cuda-toolkit libnvjpeg-dev

# Or install CUDA Toolkit from NVIDIA
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run
```

**For NCCL**:
```bash
# Ubuntu/Debian
sudo apt install libnccl-dev

# Or download from NVIDIA
wget https://developer.download.nvidia.com/compute/machine-learning/nccl/redist/nccl_2.18.3-1+cuda11.8_x86_64.txz
```

**For Gloo**:
```bash
git clone https://github.com/facebookincubator/gloo.git
cd gloo && mkdir build && cd build
cmake ..
make -j && sudo make install
```

---

## Examples

### Complete GPU Training Example

```python
#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

import sys
sys.path.insert(0, 'build/python')
import turboloader

# Check GPU availability
if not turboloader.gpu_available():
    print("ERROR: GPU not available")
    exit(1)

# Create GPU pipeline
pipeline = turboloader.Pipeline(
    tar_paths=['/data/imagenet_train.tar'],
    num_workers=8,
    decode_jpeg=True,
    gpu_decode=True,  # GPU JPEG decode
    device_id=0
)

# Model
model = models.resnet50(pretrained=False).cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

# Training
for epoch in range(100):
    pipeline.start()

    while True:
        batch = pipeline.next_batch(256)
        if len(batch) == 0:
            break

        # Zero-copy: images already on GPU
        images = torch.stack([s.get_gpu_tensor() for s in batch])
        labels = get_labels(batch)  # Your label extraction

        # Training step
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    pipeline.stop()
    print(f"Epoch {epoch+1} complete")
```

### Distributed Training with 4 GPUs

```python
#!/usr/bin/env python3
"""
Launch with: torchrun --nproc_per_node=4 train_distributed.py
"""
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import sys
sys.path.insert(0, 'build/python')
import turboloader

# Initialize distributed
dist.init_process_group(backend='nccl')
rank = dist.get_rank()
world_size = dist.get_world_size()
local_rank = int(os.environ['LOCAL_RANK'])

torch.cuda.set_device(local_rank)

# Create distributed pipeline
pipeline = turboloader.DistributedPipeline(
    tar_paths=['/data/imagenet_train.tar'],
    rank=rank,
    world_size=world_size,
    local_rank=local_rank,
    num_workers=4,
    decode_jpeg=True,
    gpu_decode=True,
    shuffle=True
)

# Model with DDP
model = models.resnet50().cuda(local_rank)
model = DDP(model, device_ids=[local_rank])

# Training
for epoch in range(100):
    pipeline.start()

    while True:
        batch = pipeline.next_batch(64)  # 64 per GPU = 256 global batch
        if len(batch) == 0:
            break

        images = torch.stack([s.get_gpu_tensor() for s in batch])
        labels = get_labels(batch)

        # DDP handles gradient synchronization
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    pipeline.stop()

    # Synchronize all ranks
    dist.barrier()

    if rank == 0:
        print(f"Epoch {epoch+1} complete")

dist.destroy_process_group()
```

---

## Benchmarking

### GPU Decode Benchmark

```bash
# Compare GPU vs CPU decode
python benchmarks/gpu_decode_benchmark.py /data/imagenet.tar \
    --batch-size 64 \
    --num-workers 4

# Compare with DALI
python benchmarks/gpu_decode_benchmark.py /data/imagenet.tar \
    --batch-size 64
```

### Distributed Benchmark

```bash
# 4 GPUs on single node
torchrun --nproc_per_node=4 benchmarks/distributed_benchmark.py /data/imagenet.tar

# 8 GPUs across 2 nodes
# Node 0:
torchrun --nnodes=2 --node_rank=0 --master_addr=192.168.1.1 \
         --nproc_per_node=4 benchmarks/distributed_benchmark.py /data/imagenet.tar

# Node 1:
torchrun --nnodes=2 --node_rank=1 --master_addr=192.168.1.1 \
         --nproc_per_node=4 benchmarks/distributed_benchmark.py /data/imagenet.tar
```

---

## Troubleshooting

### GPU Decoder Not Available

**Error**: `GPU decoder not available`

**Solutions**:
1. Check CUDA installation: `nvidia-smi`
2. Verify TurboLoader compiled with CUDA: `cmake .. | grep CUDA`
3. Check nvJPEG library: `ldconfig -p | grep nvjpeg`

### NCCL Initialization Failed

**Error**: `NCCL initialization failed`

**Solutions**:
1. Check NCCL version: `dpkg -l | grep nccl`
2. Verify network connectivity between nodes
3. Check firewall rules for master_port (default: 29500)
4. Set environment: `export NCCL_DEBUG=INFO`

### Out of GPU Memory

**Error**: `CUDA out of memory`

**Solutions**:
1. Reduce batch size
2. Reduce number of workers
3. Disable GPU decoding for some batches
4. Use gradient checkpointing in model

---

## See Also

- [API Documentation](API.md) - Complete API reference
- [Performance Tuning](PERFORMANCE.md) - Optimization guide
- [Benchmarks](../benchmarks/README.md) - Performance comparisons
- [NVIDIA nvJPEG Docs](https://docs.nvidia.com/cuda/nvjpeg/)
- [NCCL Docs](https://docs.nvidia.com/deeplearning/nccl/)
