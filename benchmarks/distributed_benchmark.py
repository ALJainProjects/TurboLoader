#!/usr/bin/env python3
"""
Distributed Training Benchmark: Multi-GPU Data Loading

Benchmarks data loading performance in distributed training scenarios.

Requirements:
- Multiple NVIDIA GPUs
- TurboLoader compiled with -DTURBOLOADER_WITH_NCCL=ON
- PyTorch with distributed support

Usage:
    # Single-node, multi-GPU (recommended for testing)
    torchrun --nproc_per_node=4 distributed_benchmark.py /path/to/dataset.tar

    # Multi-node (2 nodes, 4 GPUs each)
    # Node 0:
    torchrun --nnodes=2 --node_rank=0 --master_addr=192.168.1.1 --master_port=29500 \
             --nproc_per_node=4 distributed_benchmark.py /path/to/dataset.tar
    # Node 1:
    torchrun --nnodes=2 --node_rank=1 --master_addr=192.168.1.1 --master_port=29500 \
             --nproc_per_node=4 distributed_benchmark.py /path/to/dataset.tar
"""

import argparse
import time
import os
import sys

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Try importing TurboLoader distributed
try:
    sys.path.insert(0, 'build/python')
    import turboloader
    TURBOLOADER_AVAILABLE = True
except ImportError:
    print("Warning: TurboLoader not available")
    TURBOLOADER_AVAILABLE = False


def setup_distributed():
    """Initialize distributed process group"""
    # Get distributed environment variables
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    # Initialize process group
    dist.init_process_group(backend='nccl', init_method='env://')

    # Set device
    torch.cuda.set_device(local_rank)

    return rank, local_rank, world_size


def cleanup_distributed():
    """Cleanup distributed process group"""
    dist.destroy_process_group()


def benchmark_turboloader_distributed(tar_path, batch_size, num_workers, rank, local_rank, world_size, num_epochs=3):
    """Benchmark TurboLoader distributed data loading"""
    if not TURBOLOADER_AVAILABLE:
        if rank == 0:
            print("  [SKIP] TurboLoader not available")
        return None

    if rank == 0:
        print(f"\n=== TurboLoader Distributed ===")
        print(f"  World size: {world_size}, Batch size: {batch_size}, Workers: {num_workers}")

    # Create distributed pipeline
    pipeline = turboloader.DistributedPipeline(
        tar_paths=[tar_path],
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        num_workers=num_workers,
        decode_jpeg=True,
        gpu_decode=True,  # Use GPU decode if available
        shuffle=True
    )

    total_samples = 0
    total_time = 0.0

    for epoch in range(num_epochs):
        pipeline.start()

        # Synchronize all ranks before timing
        dist.barrier()
        epoch_start = time.time()
        epoch_samples = 0

        while True:
            batch = pipeline.next_batch(batch_size)
            if len(batch) == 0:
                break

            # Simulate processing
            epoch_samples += len(batch)

        # Synchronize all ranks after epoch
        dist.barrier()
        epoch_time = time.time() - epoch_start

        # Gather statistics from all ranks
        epoch_samples_tensor = torch.tensor(epoch_samples, device=f'cuda:{local_rank}')
        dist.all_reduce(epoch_samples_tensor, op=dist.ReduceOp.SUM)
        total_epoch_samples = epoch_samples_tensor.item()

        total_samples += total_epoch_samples
        total_time += epoch_time

        throughput = total_epoch_samples / epoch_time

        if rank == 0:
            print(f"  Epoch {epoch+1}: {throughput:.2f} img/s "
                  f"({total_epoch_samples} images in {epoch_time:.2f}s)")

        pipeline.stop()

    avg_throughput = total_samples / total_time

    if rank == 0:
        print(f"  Average: {avg_throughput:.2f} img/s (across {world_size} GPUs)")
        print(f"  Per-GPU: {avg_throughput/world_size:.2f} img/s")

    return {
        'framework': 'TurboLoader (Distributed)',
        'throughput': avg_throughput,
        'per_gpu_throughput': avg_throughput / world_size,
        'batch_size': batch_size,
        'num_workers': num_workers,
        'world_size': world_size
    }


def benchmark_pytorch_distributed(tar_path, batch_size, num_workers, rank, local_rank, world_size, num_epochs=3):
    """Benchmark PyTorch DistributedDataParallel"""
    if rank == 0:
        print(f"\n=== PyTorch DDP ===")
        print(f"  World size: {world_size}, Batch size: {batch_size}, Workers: {num_workers}")

    # Create dataset and distributed sampler
    from torch.utils.data import Dataset, DataLoader
    from torch.utils.data.distributed import DistributedSampler
    import tarfile

    class TarDataset(Dataset):
        def __init__(self, tar_path):
            self.tar_path = tar_path
            with tarfile.open(tar_path, 'r') as tar:
                self.members = [m for m in tar.getmembers() if m.isfile() and m.name.endswith('.jpg')]

        def __len__(self):
            return len(self.members)

        def __getitem__(self, idx):
            with tarfile.open(self.tar_path, 'r') as tar:
                member = self.members[idx]
                data = tar.extractfile(member).read()
                # Simulate decode (just return bytes for fair comparison)
                return data

    dataset = TarDataset(tar_path)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)

    total_samples = 0
    total_time = 0.0

    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)

        # Synchronize all ranks before timing
        dist.barrier()
        epoch_start = time.time()
        epoch_samples = 0

        for batch in dataloader:
            epoch_samples += len(batch)

        # Synchronize all ranks after epoch
        dist.barrier()
        epoch_time = time.time() - epoch_start

        # Gather statistics from all ranks
        epoch_samples_tensor = torch.tensor(epoch_samples, device=f'cuda:{local_rank}')
        dist.all_reduce(epoch_samples_tensor, op=dist.ReduceOp.SUM)
        total_epoch_samples = epoch_samples_tensor.item()

        total_samples += total_epoch_samples
        total_time += epoch_time

        throughput = total_epoch_samples / epoch_time

        if rank == 0:
            print(f"  Epoch {epoch+1}: {throughput:.2f} img/s "
                  f"({total_epoch_samples} images in {epoch_time:.2f}s)")

    avg_throughput = total_samples / total_time

    if rank == 0:
        print(f"  Average: {avg_throughput:.2f} img/s (across {world_size} GPUs)")
        print(f"  Per-GPU: {avg_throughput/world_size:.2f} img/s")

    return {
        'framework': 'PyTorch DDP',
        'throughput': avg_throughput,
        'per_gpu_throughput': avg_throughput / world_size,
        'batch_size': batch_size,
        'num_workers': num_workers,
        'world_size': world_size
    }


def print_summary(results, rank):
    """Print comparison summary (rank 0 only)"""
    if rank != 0:
        return

    print("\n" + "="*80)
    print("DISTRIBUTED TRAINING BENCHMARK SUMMARY")
    print("="*80)

    if not results:
        print("No results to display")
        return

    # Sort by total throughput
    results = sorted(results, key=lambda x: x['throughput'], reverse=True)

    print(f"\n{'Framework':<25} {'Total':<15} {'Per-GPU':<15} {'GPUs':<8} {'Batch':<8}")
    print("-"*80)

    for r in results:
        print(f"{r['framework']:<25} {r['throughput']:>10.2f} img/s  "
              f"{r['per_gpu_throughput']:>10.2f} img/s  "
              f"{r['world_size']:<8} {r['batch_size']:<8}")

    if len(results) > 1:
        print("\nSpeedup:")
        baseline = results[-1]['throughput']
        for r in results:
            speedup = r['throughput'] / baseline
            print(f"  - {r['framework']}: {speedup:.2f}x")


def main():
    parser = argparse.ArgumentParser(description='Distributed Training Benchmark')
    parser.add_argument('tar_path', help='Path to TAR dataset')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size per GPU (default: 32)')
    parser.add_argument('--num-workers', type=int, default=4, help='Workers per GPU (default: 4)')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs (default: 3)')
    parser.add_argument('--skip-turboloader', action='store_true', help='Skip TurboLoader')
    parser.add_argument('--skip-pytorch', action='store_true', help='Skip PyTorch DDP')

    args = parser.parse_args()

    # Setup distributed
    rank, local_rank, world_size = setup_distributed()

    if rank == 0:
        print("="*80)
        print("DISTRIBUTED TRAINING DATA LOADING BENCHMARK")
        print("="*80)
        print(f"Dataset: {args.tar_path}")
        print(f"World size: {world_size} GPUs")
        print(f"Batch size per GPU: {args.batch_size}")
        print(f"Workers per GPU: {args.num_workers}")
        print(f"Epochs: {args.epochs}")
        print("="*80)

    results = []

    # Benchmark TurboLoader distributed
    if not args.skip_turboloader:
        result = benchmark_turboloader_distributed(
            args.tar_path,
            args.batch_size,
            args.num_workers,
            rank,
            local_rank,
            world_size,
            args.epochs
        )
        if result:
            results.append(result)

    # Benchmark PyTorch DDP
    if not args.skip_pytorch:
        result = benchmark_pytorch_distributed(
            args.tar_path,
            args.batch_size,
            args.num_workers,
            rank,
            local_rank,
            world_size,
            args.epochs
        )
        if result:
            results.append(result)

    # Print summary
    print_summary(results, rank)

    # Cleanup
    cleanup_distributed()


if __name__ == '__main__':
    main()
