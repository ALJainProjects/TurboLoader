# Distributed Training Guide

Complete guide for using TurboLoader with distributed training frameworks.

## Overview

TurboLoader provides automatic data sharding for distributed training with:
- **Deterministic sharding** - Each rank gets non-overlapping data
- **Equal batch sizes** - Ensures synchronized training across GPUs
- **Zero configuration** - Works out-of-the-box with PyTorch DDP, Horovod, DeepSpeed

---

## PyTorch DistributedDataParallel (DDP)

### Quick Start

Enable distributed training with one parameter:

```python
import turboloader
import torch.distributed as dist

# Initialize distributed training
dist.init_process_group(backend='nccl')

# Create DataLoader with automatic sharding
loader = turboloader.DataLoader(
    'imagenet_train.tar',
    batch_size=256,  # Per-GPU batch size
    num_workers=8,
    shuffle=True,
    enable_distributed=True,  # Enable automatic sharding
    drop_last=True            # Ensure equal batches across GPUs
)

# Each rank will get different data automatically
for batch in loader:
    # Process batch on this GPU
    ...
```

### Complete DDP Example

See [examples/distributed_ddp.py](../examples/distributed_ddp.py) for a production-ready example.

Key features:
- Multi-GPU automatic data sharding
- Synchronized metrics across ranks
- Model checkpointing (rank 0 only)
- Learning rate scaling with world size

**Usage:**
```bash
# Single machine, 4 GPUs
python examples/distributed_ddp.py \
    --data-path imagenet.tar \
    --gpus 4 \
    --batch-size 256

# Multi-node (requires MASTER_ADDR and MASTER_PORT)
python examples/distributed_ddp.py \
    --data-path imagenet.tar \
    --gpus 8 \
    --nodes 2 \
    --node-rank 0
```

### How Data Sharding Works

TurboLoader automatically shards data based on world rank and size:

```python
# Rank 0 gets samples 0, 4, 8, 12, ...
# Rank 1 gets samples 1, 5, 9, 13, ...
# Rank 2 gets samples 2, 6, 10, 14, ...
# Rank 3 gets samples 3, 7, 11, 15, ...
```

This ensures:
- No data overlap between ranks
- Deterministic data distribution
- Efficient memory usage (no duplication)

---

## Multi-Node Training

### Prerequisites

1. **Network Connectivity**: All nodes must be able to communicate
2. **Shared Storage**: Data accessible from all nodes (NFS, S3, etc.)
3. **Same Environment**: Same Python packages and TurboLoader version

### Setup

**Node 0 (Master):**
```bash
export MASTER_ADDR=192.168.1.100  # IP of master node
export MASTER_PORT=12355          # Any free port

python examples/distributed_ddp.py \
    --data-path /shared/imagenet.tar \
    --gpus 4 \
    --nodes 2 \
    --node-rank 0
```

**Node 1 (Worker):**
```bash
export MASTER_ADDR=192.168.1.100  # Same as master
export MASTER_PORT=12355          # Same port

python examples/distributed_ddp.py \
    --data-path /shared/imagenet.tar \
    --gpus 4 \
    --nodes 2 \
    --node-rank 1
```

### Data Access Patterns

**Option 1: Shared Storage (Recommended)**
```python
# All nodes access same TAR file via NFS
loader = turboloader.DataLoader(
    '/shared/imagenet.tar',  # NFS mount
    batch_size=256,
    num_workers=8,
    enable_distributed=True
)
```

**Option 2: Local Copies**
```python
# Each node has local copy of data
loader = turboloader.DataLoader(
    f'/local/node{node_rank}/imagenet.tar',
    batch_size=256,
    num_workers=8,
    enable_distributed=True
)
```

**Option 3: Cloud Storage**
```python
# Load from S3
loader = turboloader.DataLoader(
    's3://bucket/imagenet.tar',
    batch_size=256,
    num_workers=8,
    enable_distributed=True
)
```

---

## SLURM Integration

Run distributed training on SLURM clusters:

### SLURM Script

```bash
#!/bin/bash
#SBATCH --job-name=turboloader-train
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4  # GPUs per node
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00

# Load modules
module load python/3.11
module load cuda/11.8

# Set environment
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=12355

# Run training
srun python examples/distributed_ddp.py \
    --data-path /shared/imagenet.tar \
    --gpus 4 \
    --nodes $SLURM_NNODES \
    --node-rank $SLURM_NODEID \
    --batch-size 256
```

Submit job:
```bash
sbatch train.slurm
```

---

## Kubernetes / Cloud

### Kubernetes with Kubeflow

```yaml
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: turboloader-training
spec:
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      template:
        spec:
          containers:
          - name: pytorch
            image: my-turboloader-image:latest
            command:
            - python
            - examples/distributed_ddp.py
            - --data-path
            - /data/imagenet.tar
            - --gpus
            - "4"
    Worker:
      replicas: 3
      template:
        spec:
          containers:
          - name: pytorch
            image: my-turboloader-image:latest
            command:
            - python
            - examples/distributed_ddp.py
            - --data-path
            - /data/imagenet.tar
            - --gpus
            - "4"
```

---

## Horovod Integration

TurboLoader works with Horovod:

```python
import turboloader
import horovod.torch as hvd

# Initialize Horovod
hvd.init()

# Pin GPU
torch.cuda.set_device(hvd.local_rank())

# Create DataLoader (TurboLoader detects Horovod automatically)
loader = turboloader.DataLoader(
    'imagenet.tar',
    batch_size=256,
    num_workers=8,
    shuffle=True,
    enable_distributed=True  # Works with Horovod
)

# Model with Horovod DDP
model = torch.nn.Sequential(...)
model = model.cuda()

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
optimizer = hvd.DistributedOptimizer(optimizer)

# Broadcast parameters
hvd.broadcast_parameters(model.state_dict(), root_rank=0)

# Training loop
for batch in loader:
    # Process batch
    ...
```

Run with Horovod:
```bash
horovodrun -np 4 python train.py
```

---

## DeepSpeed Integration

Use TurboLoader with DeepSpeed:

```python
import turboloader
import deepspeed

# Create DataLoader
loader = turboloader.DataLoader(
    'imagenet.tar',
    batch_size=256,
    num_workers=8,
    enable_distributed=True
)

# Initialize DeepSpeed
model, optimizer, _, _ = deepspeed.initialize(
    model=model,
    optimizer=optimizer,
    config=ds_config
)

# Training loop
for batch in loader:
    # Process batch
    loss = model(batch)
    model.backward(loss)
    model.step()
```

---

## Best Practices

### 1. Use drop_last=True

Ensure equal batch sizes across all ranks:

```python
loader = turboloader.DataLoader(
    'data.tar',
    batch_size=256,
    enable_distributed=True,
    drop_last=True  # Critical for distributed training
)
```

### 2. Scale Learning Rate

Scale learning rate with world size:

```python
import torch.distributed as dist

world_size = dist.get_world_size()
base_lr = 0.1

# Linear scaling rule
scaled_lr = base_lr * world_size

optimizer = torch.optim.SGD(model.parameters(), lr=scaled_lr)
```

### 3. Synchronize Metrics

Reduce metrics across all ranks:

```python
import torch.distributed as dist

# Local metrics
local_loss = torch.tensor(loss.item()).cuda()
local_acc = torch.tensor(accuracy).cuda()

# Reduce across all ranks
dist.all_reduce(local_loss, op=dist.ReduceOp.AVG)
dist.all_reduce(local_acc, op=dist.ReduceOp.AVG)

# Now local_loss and local_acc contain global averages
```

### 4. Save Checkpoints on Rank 0 Only

```python
import torch.distributed as dist

if dist.get_rank() == 0:
    torch.save(model.state_dict(), 'checkpoint.pt')
    print("Checkpoint saved")

# Synchronize before continuing
dist.barrier()
```

### 5. Set Worker Count

Use fewer workers per GPU to avoid CPU saturation:

```python
# Single GPU: 8 workers
# 4 GPUs: 4 workers per GPU (16 total)
# 8 GPUs: 2 workers per GPU (16 total)

gpus = 4
workers_per_gpu = max(2, 16 // gpus)

loader = turboloader.DataLoader(
    'data.tar',
    batch_size=256,
    num_workers=workers_per_gpu
)
```

---

## Troubleshooting

### Issue: Different ranks see same data

**Problem**: `enable_distributed` not set.

**Solution:**
```python
loader = turboloader.DataLoader(
    'data.tar',
    enable_distributed=True  # Must be True!
)
```

### Issue: Unequal batch sizes across ranks

**Problem**: `drop_last` not set.

**Solution:**
```python
loader = turboloader.DataLoader(
    'data.tar',
    enable_distributed=True,
    drop_last=True  # Critical!
)
```

### Issue: Training hangs

**Problem**: Ranks waiting for each other.

**Solution:** Ensure all ranks call `dist.barrier()` at same points:

```python
for epoch in range(epochs):
    # Train
    for batch in loader:
        ...

    # Synchronize after epoch
    dist.barrier()
```

### Issue: Out of memory on some ranks

**Problem**: Data distribution is uneven.

**Solution:** Use `drop_last=True` and ensure equal batch sizes.

---

## Performance Tips

### 1. Data Prefetching

Enable prefetching for better GPU utilization:

```python
# TurboLoader automatically prefetches
# Adjust workers for optimal performance
loader = turboloader.DataLoader(
    'data.tar',
    batch_size=256,
    num_workers=8  # Tune this
)
```

### 2. Gradient Accumulation

Train with larger effective batch size:

```python
accumulation_steps = 4

for i, batch in enumerate(loader):
    loss = model(batch) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 3. Use Fast Storage

- **Best**: Local NVMe SSD
- **Good**: Network storage with high bandwidth (100+ Gbps)
- **OK**: S3 with good network connection
- **Slow**: Network storage with low bandwidth

---

## Examples

Complete examples available:

1. **[PyTorch DDP](../examples/distributed_ddp.py)** - Multi-GPU training with DDP
2. **[ImageNet ResNet50](../examples/imagenet_resnet50.py)** - Full training pipeline with DDP
3. **[PyTorch Lightning](../examples/pytorch_lightning_example.py)** - Lightning with distributed support

---

## Additional Resources

- [PyTorch Integration Guide](guides/pytorch-integration.md)
- [ImageNet Training Example](../examples/imagenet_resnet50.py)
- [PyTorch DDP Documentation](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [SLURM Documentation](https://slurm.schedmd.com/)
