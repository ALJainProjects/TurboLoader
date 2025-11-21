# PyTorch Integration Guide

Complete guide for integrating TurboLoader with PyTorch for training, validation, and inference.

## Table of Contents

- [Quick Start](#quick-start)
- [Basic Integration](#basic-integration)
- [Training Pipeline](#training-pipeline)
- [Distributed Training](#distributed-training)
- [PyTorch Lightning](#pytorch-lightning)
- [Best Practices](#best-practices)
- [Performance Tuning](#performance-tuning)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

Replace PyTorch DataLoader with TurboLoader in 3 steps:

```python
import turboloader
import torch

# 1. Create transforms
transforms = turboloader.Compose([
    turboloader.Resize(224, 224),
    turboloader.ImageNetNormalize(),
    turboloader.ToTensor()
])

# 2. Create TurboLoader
loader = turboloader.DataLoader(
    'imagenet_train.tar',
    batch_size=256,
    num_workers=8,
    shuffle=True
)

# 3. Training loop
for batch in loader:
    images = torch.stack([
        torch.from_numpy(transforms.apply(s['image'])).float()
        for s in batch
    ])
    labels = torch.tensor([s['label'] for s in batch])

    # Train your model
    outputs = model(images)
    loss = criterion(outputs, labels)
    # ...
```

---

## Basic Integration

### Convert NumPy to PyTorch Tensors

TurboLoader returns NumPy arrays. Convert to PyTorch tensors:

```python
import turboloader
import torch
import numpy as np

loader = turboloader.DataLoader('data.tar', batch_size=32, num_workers=4)

for batch in loader:
    for sample in batch:
        # NumPy array (H, W, C) uint8
        image_np = sample['image']

        # Convert to PyTorch tensor (C, H, W) float32
        image_tensor = torch.from_numpy(image_np)

        # Permute from HWC to CHW
        image_tensor = image_tensor.permute(2, 0, 1).float()

        # Normalize to [0, 1]
        image_tensor /= 255.0
```

### Using ToTensor Transform

Simplify with `ToTensor` transform:

```python
to_tensor = turboloader.ToTensor(turboloader.TensorFormat.PYTORCH_CHW)

for batch in loader:
    for sample in batch:
        # Automatically converts to CHW format
        image_chw = to_tensor.apply(sample['image'])  # (C, H, W)

        # Convert NumPy to PyTorch tensor
        image_tensor = torch.from_numpy(image_chw).float() / 255.0
```

---

## Training Pipeline

### Complete ImageNet Training Example

```python
import turboloader
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model
model = resnet50(num_classes=1000).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

# Training transforms
train_transforms = turboloader.Compose([
    turboloader.Resize(256, 256),
    turboloader.RandomCrop(224, 224),
    turboloader.RandomHorizontalFlip(0.5),
    turboloader.ColorJitter(0.2, 0.2, 0.2, 0.1),
    turboloader.ImageNetNormalize(),
    turboloader.ToTensor()
])

# Create DataLoader
train_loader = turboloader.DataLoader(
    'imagenet_train.tar',
    batch_size=256,
    num_workers=8,
    shuffle=True
)

# Training loop
model.train()
for epoch in range(90):
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, batch in enumerate(train_loader):
        # Process batch
        images = []
        labels = []

        for sample in batch:
            img = train_transforms.apply(sample['image'])
            images.append(torch.from_numpy(img).float())
            labels.append(sample['label'])

        # Stack into batch tensors
        images = torch.stack(images).to(device)
        labels = torch.tensor(labels, dtype=torch.long).to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if batch_idx % 100 == 0:
            print(f'Epoch {epoch} [{batch_idx}/{len(train_loader)}] '
                  f'Loss: {running_loss/(batch_idx+1):.3f} '
                  f'Acc: {100.*correct/total:.2f}%')

    print(f'Epoch {epoch} completed. Acc: {100.*correct/total:.2f}%')
```

### Validation Pipeline

Use deterministic transforms for validation:

```python
# Validation transforms (no randomness)
val_transforms = turboloader.Compose([
    turboloader.Resize(256, 256),
    turboloader.CenterCrop(224, 224),
    turboloader.ImageNetNormalize(),
    turboloader.ToTensor()
])

val_loader = turboloader.DataLoader(
    'imagenet_val.tar',
    batch_size=256,
    num_workers=8,
    shuffle=False  # Don't shuffle validation
)

# Validation loop
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for batch in val_loader:
        images = []
        labels = []

        for sample in batch:
            img = val_transforms.apply(sample['image'])
            images.append(torch.from_numpy(img).float())
            labels.append(sample['label'])

        images = torch.stack(images).to(device)
        labels = torch.tensor(labels, dtype=torch.long).to(device)

        outputs = model(images)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

print(f'Validation Accuracy: {100.*correct/total:.2f}%')
```

---

## Distributed Training

### PyTorch DDP (DistributedDataParallel)

TurboLoader supports automatic data sharding for multi-GPU training:

```python
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import turboloader

def train(rank, world_size):
    # Initialize process group
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )

    # Set device
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    # Model
    model = resnet50(num_classes=1000).to(device)
    model = DDP(model, device_ids=[rank])

    # TurboLoader with distributed sharding
    loader = turboloader.DataLoader(
        'imagenet_train.tar',
        batch_size=256,  # Per-GPU batch size
        num_workers=8,
        shuffle=True,
        enable_distributed=True,  # Enable automatic sharding
        drop_last=True  # Ensure equal batches across GPUs
    )

    # Transforms
    transforms = turboloader.Compose([
        turboloader.Resize(256, 256),
        turboloader.RandomCrop(224, 224),
        turboloader.RandomHorizontalFlip(0.5),
        turboloader.ImageNetNormalize(),
        turboloader.ToTensor()
    ])

    # Training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    model.train()
    for epoch in range(90):
        for batch in loader:
            images = []
            labels = []

            for sample in batch:
                img = transforms.apply(sample['image'])
                images.append(torch.from_numpy(img).float())
                labels.append(sample['label'])

            images = torch.stack(images).to(device)
            labels = torch.tensor(labels, dtype=torch.long).to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        if rank == 0:
            print(f'Epoch {epoch} completed')

    dist.destroy_process_group()

if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
```

### Key Points for Distributed Training

1. **Enable distributed mode**: Set `enable_distributed=True`
2. **Drop last batch**: Use `drop_last=True` for equal batch sizes
3. **Per-GPU batch size**: Batch size is per process, not global
4. **Deterministic sharding**: Each rank gets non-overlapping data shard

---

## PyTorch Lightning

See [examples/pytorch_lightning_example.py](../../examples/pytorch_lightning_example.py) for full example.

### LightningDataModule

```python
import pytorch_lightning as pl
from torch.utils.data import IterableDataset, DataLoader as TorchDataLoader
import turboloader

class TurboLoaderWrapper(IterableDataset):
    def __init__(self, data_path, batch_size, num_workers, transform=None):
        self.loader = turboloader.DataLoader(
            data_path,
            batch_size=batch_size,
            num_workers=num_workers
        )
        self.transform = transform

    def __iter__(self):
        for batch in self.loader:
            for sample in batch:
                img = sample['image']
                if self.transform:
                    img = self.transform.apply(img)

                img_tensor = torch.from_numpy(img).float()
                label = sample.get('label', 0)

                yield img_tensor, label

class ImageNetDataModule(pl.LightningDataModule):
    def __init__(self, train_path, val_path, batch_size=256, num_workers=8):
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_transform = turboloader.Compose([
            turboloader.Resize(256, 256),
            turboloader.RandomCrop(224, 224),
            turboloader.RandomHorizontalFlip(0.5),
            turboloader.ImageNetNormalize(),
            turboloader.ToTensor()
        ])

        self.val_transform = turboloader.Compose([
            turboloader.Resize(256, 256),
            turboloader.CenterCrop(224, 224),
            turboloader.ImageNetNormalize(),
            turboloader.ToTensor()
        ])

    def train_dataloader(self):
        dataset = TurboLoaderWrapper(
            self.train_path,
            self.batch_size,
            self.num_workers,
            self.train_transform
        )
        return TorchDataLoader(dataset, batch_size=None, num_workers=0)

    def val_dataloader(self):
        dataset = TurboLoaderWrapper(
            self.val_path,
            self.batch_size,
            self.num_workers,
            self.val_transform
        )
        return TorchDataLoader(dataset, batch_size=None, num_workers=0)
```

---

## Best Practices

### 1. Transform Ordering

Apply transforms in correct order:

```python
transforms = turboloader.Compose([
    # 1. Resize first
    turboloader.Resize(256, 256),

    # 2. Spatial augmentations
    turboloader.RandomCrop(224, 224),
    turboloader.RandomHorizontalFlip(0.5),

    # 3. Color augmentations
    turboloader.ColorJitter(0.2, 0.2, 0.2, 0.1),

    # 4. Normalize last
    turboloader.ImageNetNormalize(),

    # 5. Convert to tensor format
    turboloader.ToTensor()
])
```

### 2. Pin Memory for Faster GPU Transfer

```python
# Convert batch to pinned memory for faster GPU transfer
images = torch.stack(images)
images = images.pin_memory().to(device, non_blocking=True)
```

### 3. Mixed Precision Training

Use TurboLoader with PyTorch AMP:

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in loader:
    images = process_batch(batch)  # Your processing
    images = images.to(device)

    optimizer.zero_grad()

    with autocast():
        outputs = model(images)
        loss = criterion(outputs, labels)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### 4. Gradient Accumulation

For large models with small batch sizes:

```python
accumulation_steps = 4

for batch_idx, batch in enumerate(loader):
    images, labels = process_batch(batch)
    images, labels = images.to(device), labels.to(device)

    outputs = model(images)
    loss = criterion(outputs, labels) / accumulation_steps

    loss.backward()

    if (batch_idx + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

## Performance Tuning

### Optimal Worker Count

Benchmark different worker counts:

```python
import time

for num_workers in [1, 2, 4, 8, 16]:
    loader = turboloader.DataLoader(
        'data.tar',
        batch_size=256,
        num_workers=num_workers
    )

    start = time.time()
    count = 0
    for batch in loader:
        count += len(batch)
        if count >= 10000:
            break

    elapsed = time.time() - start
    throughput = count / elapsed
    print(f'Workers: {num_workers}, Throughput: {throughput:.1f} img/s')
```

### Batch Size Tuning

Larger batches = better throughput:

```python
# Too small - underutilizes GPU
batch_size = 32  # ❌

# Good - balances memory and throughput
batch_size = 256  # ✓

# Adjust based on GPU memory
batch_size = 512  # For A100 80GB
```

### Prefetch to GPU

Overlap data loading with computation:

```python
# Prefetch next batch while training current
import threading
import queue

def prefetch_worker(loader, queue, device):
    for batch in loader:
        images = process_batch(batch)
        images = images.to(device, non_blocking=True)
        queue.put(images)
    queue.put(None)  # Sentinel

q = queue.Queue(maxsize=2)  # Prefetch 2 batches
thread = threading.Thread(target=prefetch_worker, args=(loader, q, device))
thread.start()

while True:
    images = q.get()
    if images is None:
        break
    # Train on images
    outputs = model(images)
    # ...
```

---

## Troubleshooting

### Issue: Slow Data Loading

**Symptoms**: GPU utilization < 80%, training takes too long

**Solutions**:
1. Increase `num_workers` (try 8, 16)
2. Increase `batch_size` if GPU memory allows
3. Use TBL format instead of TAR for faster loading
4. Check if CPU is bottleneck with `htop`

### Issue: Out of Memory

**Symptoms**: `CUDA out of memory` error

**Solutions**:
1. Reduce `batch_size`
2. Use gradient accumulation
3. Enable mixed precision training
4. Reduce image resolution

### Issue: Inconsistent Validation Results

**Symptoms**: Validation accuracy varies between runs

**Solutions**:
1. Use `CenterCrop` instead of `RandomCrop` for validation
2. Set `shuffle=False` for validation loader
3. Disable random transforms (RandomFlip, ColorJitter, etc.)

### Issue: Data Mismatch in Distributed Training

**Symptoms**: Different GPUs see same data

**Solutions**:
1. Set `enable_distributed=True` in DataLoader
2. Initialize distributed process group before creating DataLoader
3. Use `drop_last=True` for equal batch sizes

---

## Additional Resources

- [Transforms API Reference](../api/transforms.md) - Complete transform documentation
- [PyTorch Lightning Example](../../examples/pytorch_lightning_example.py) - Full working example
- [Distributed Training Guide](distributed.md) - Multi-node setup
- [Performance Benchmarks](../benchmarks/index.md) - Throughput analysis

---

## Example: Complete Training Script

See [examples/imagenet_resnet50.py](../../examples/imagenet_resnet50.py) for a complete, production-ready training script with:
- Multi-GPU DDP training
- Learning rate scheduling
- Checkpointing
- TensorBoard logging
- Mixed precision training
