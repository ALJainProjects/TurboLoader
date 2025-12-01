#!/usr/bin/env python3
"""
PyTorch DistributedDataParallel (DDP) Example with TurboLoader

This example demonstrates multi-GPU training with PyTorch DDP and TurboLoader.

Features:
- Automatic data sharding across GPUs
- Synchronized batch normalization
- Gradient synchronization
- Model checkpointing on rank 0
- Per-GPU metrics logging

Requirements:
    pip install torch torchvision turboloader

Usage:
    # Single machine, multiple GPUs
    python distributed_ddp.py --data-path /path/to/data.tar --gpus 4

    # Multi-node (requires MASTER_ADDR and MASTER_PORT environment variables)
    python distributed_ddp.py --data-path /path/to/data.tar --gpus 8 --nodes 2 --node-rank 0
"""

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    from torchvision.models import resnet50
except ImportError:
    print("Error: torchvision not installed")
    print("Install with: pip install torchvision")
    exit(1)

import turboloader


def setup_dist(rank, world_size):
    """Initialize distributed training environment."""
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "12355")

    # Initialize process group
    dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)

    # Set device
    torch.cuda.set_device(rank)


def cleanup_dist():
    """Clean up distributed training."""
    dist.destroy_process_group()


def train_worker(rank, world_size, args):
    """Training worker function for each GPU."""
    print(f"[Rank {rank}] Starting training worker")

    # Setup distributed training
    setup_dist(rank, world_size)

    # Set device
    device = torch.device(f"cuda:{rank}")

    # Create model
    model = resnet50(num_classes=args.num_classes)
    model = model.to(device)

    # Wrap with DDP
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr * world_size,  # Scale learning rate with world size
        momentum=0.9,
        weight_decay=1e-4,
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Create TurboLoader with distributed sharding
    loader = turboloader.DataLoader(
        args.data_path,
        batch_size=args.batch_size,  # Per-GPU batch size
        num_workers=args.num_workers,
        shuffle=True,
        enable_distributed=True,  # Enable automatic sharding
        drop_last=True,  # Ensure equal batches across GPUs
    )

    # Create transforms
    transforms = turboloader.Compose(
        [
            turboloader.Resize(256, 256),
            turboloader.RandomCrop(224, 224),
            turboloader.RandomHorizontalFlip(0.5),
            turboloader.ColorJitter(0.2, 0.2, 0.2, 0.1),
            turboloader.ImageNetNormalize(),
            turboloader.ToTensor(),
        ]
    )

    # Training loop
    for epoch in range(args.epochs):
        model.train()

        epoch_start = time.time()
        running_loss = 0.0
        correct = 0
        total = 0
        num_batches = 0

        for batch_idx, batch in enumerate(loader):
            # Process batch
            images = []
            labels = []

            for sample in batch:
                img = transforms.apply(sample["image"])
                images.append(torch.from_numpy(img).float())
                labels.append(sample.get("label", 0))

            # Stack into batch tensors
            images = torch.stack(images).to(device, non_blocking=True)
            labels = torch.tensor(labels, dtype=torch.long).to(device, non_blocking=True)

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
            num_batches += 1

            # Log progress (only rank 0)
            if rank == 0 and batch_idx % args.log_interval == 0:
                avg_loss = running_loss / (batch_idx + 1)
                acc = 100.0 * correct / total
                print(
                    f"[Rank {rank}] Epoch {epoch} [{batch_idx}/{num_batches}] "
                    f"Loss: {avg_loss:.3f} Acc: {acc:.2f}%"
                )

        # End of epoch
        epoch_time = time.time() - epoch_start

        # Synchronize metrics across all ranks
        avg_loss = torch.tensor(running_loss / num_batches).to(device)
        acc = torch.tensor(100.0 * correct / total).to(device)

        dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
        dist.all_reduce(acc, op=dist.ReduceOp.AVG)

        # Log epoch summary (only rank 0)
        if rank == 0:
            samples_per_sec = total * world_size / epoch_time
            print(f'\n{"="*70}')
            print(f"Epoch {epoch} Summary:")
            print(f"  Loss: {avg_loss.item():.4f}")
            print(f"  Accuracy: {acc.item():.2f}%")
            print(f"  Time: {epoch_time:.2f}s")
            print(f"  Throughput: {samples_per_sec:.1f} samples/sec")
            print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
            print(f'{"="*70}\n')

        # Step scheduler
        scheduler.step()

        # Save checkpoint (only rank 0)
        if rank == 0 and (epoch + 1) % args.save_interval == 0:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": avg_loss.item(),
                "accuracy": acc.item(),
            }
            checkpoint_path = f"{args.checkpoint_dir}/checkpoint_epoch_{epoch}.pt"
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            torch.save(checkpoint, checkpoint_path)
            print(f"[Rank {rank}] Saved checkpoint: {checkpoint_path}")

        # Synchronize before next epoch
        dist.barrier()

    # Final cleanup
    if rank == 0:
        print(f'\n{"="*70}')
        print("Training completed!")
        print(f"Final accuracy: {acc.item():.2f}%")
        print(f'{"="*70}')

    cleanup_dist()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="PyTorch DDP Training with TurboLoader")
    parser.add_argument(
        "--data-path", type=str, required=True, help="Path to training data (TAR or TBL format)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=128, help="Per-GPU batch size (default: 128)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers per GPU (default: 4)",
    )
    parser.add_argument(
        "--epochs", type=int, default=90, help="Number of training epochs (default: 90)"
    )
    parser.add_argument("--lr", type=float, default=0.1, help="Base learning rate (default: 0.1)")
    parser.add_argument(
        "--num-classes", type=int, default=1000, help="Number of classes (default: 1000)"
    )
    parser.add_argument(
        "--gpus", type=int, default=None, help="Number of GPUs (default: all available)"
    )
    parser.add_argument(
        "--log-interval", type=int, default=100, help="Logging interval in batches (default: 100)"
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=10,
        help="Checkpoint save interval in epochs (default: 10)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints",
        help="Checkpoint directory (default: ./checkpoints)",
    )
    parser.add_argument("--nodes", type=int, default=1, help="Number of nodes (default: 1)")
    parser.add_argument("--node-rank", type=int, default=0, help="Node rank (default: 0)")

    args = parser.parse_args()

    # Determine world size
    if args.gpus is None:
        args.gpus = torch.cuda.device_count()

    if args.gpus == 0:
        print("Error: No CUDA devices available")
        print("This script requires GPUs for distributed training")
        exit(1)

    world_size = args.gpus * args.nodes

    print("=" * 70)
    print("PyTorch DDP Training with TurboLoader")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Data path: {args.data_path}")
    print(f"  GPUs per node: {args.gpus}")
    print(f"  Nodes: {args.nodes}")
    print(f"  World size: {world_size}")
    print(f"  Per-GPU batch size: {args.batch_size}")
    print(f"  Global batch size: {args.batch_size * world_size}")
    print(f"  Workers per GPU: {args.num_workers}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Base learning rate: {args.lr}")
    print(f"  Scaled learning rate: {args.lr * world_size}")
    print("=" * 70 + "\n")

    # Spawn training processes
    mp.spawn(train_worker, args=(world_size, args), nprocs=args.gpus, join=True)


if __name__ == "__main__":
    main()
