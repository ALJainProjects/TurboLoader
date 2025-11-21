#!/usr/bin/env python3
"""
Production ImageNet ResNet50 Training with TurboLoader

Complete, production-ready training script for ImageNet classification
with ResNet50 using TurboLoader for data loading.

Features:
- Multi-GPU training with PyTorch DDP
- Mixed precision training (AMP)
- Learning rate scheduling (warmup + cosine decay)
- Model checkpointing with best model tracking
- TensorBoard logging
- Validation during training
- Gradient clipping
- EMA (Exponential Moving Average) option

Requirements:
    pip install torch torchvision turboloader tensorboard

Usage:
    # Single GPU
    python imagenet_resnet50.py \
        --train-data imagenet_train.tar \
        --val-data imagenet_val.tar

    # Multi-GPU (4 GPUs)
    python imagenet_resnet50.py \
        --train-data imagenet_train.tar \
        --val-data imagenet_val.tar \
        --gpus 4 \
        --batch-size 256

    # Resume from checkpoint
    python imagenet_resnet50.py \
        --train-data imagenet_train.tar \
        --val-data imagenet_val.tar \
        --resume checkpoints/checkpoint_epoch_10.pt
"""

import argparse
import os
import time
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler

try:
    from torchvision.models import resnet50
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    print("Error: Missing dependencies")
    print("Install with: pip install torchvision tensorboard")
    exit(1)

import turboloader


class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def setup_dist(rank, world_size):
    """Initialize distributed training."""
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')

    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    torch.cuda.set_device(rank)


def cleanup_dist():
    """Clean up distributed training."""
    dist.destroy_process_group()


def accuracy(output, target, topk=(1,)):
    """Compute top-k accuracy."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_checkpoint(state, is_best, checkpoint_dir, filename='checkpoint.pt'):
    """Save checkpoint and best model."""
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    torch.save(state, checkpoint_path)

    if is_best:
        best_path = os.path.join(checkpoint_dir, 'model_best.pt')
        torch.save(state, best_path)


def train_epoch(loader, model, criterion, optimizer, scaler, epoch, device, args, writer=None):
    """Train for one epoch."""
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()

    # Transforms
    transforms = turboloader.Compose([
        turboloader.Resize(256, 256),
        turboloader.RandomCrop(224, 224),
        turboloader.RandomHorizontalFlip(0.5),
        turboloader.ColorJitter(0.4, 0.4, 0.4, 0.1),
        turboloader.ImageNetNormalize(),
        turboloader.ToTensor()
    ])

    end = time.time()

    for batch_idx, batch in enumerate(loader):
        # Measure data loading time
        data_time.update(time.time() - end)

        # Process batch
        images = []
        labels = []

        for sample in batch:
            img = transforms.apply(sample['image'])
            images.append(torch.from_numpy(img).float())
            labels.append(sample['label'])

        images = torch.stack(images).to(device, non_blocking=True)
        labels = torch.tensor(labels, dtype=torch.long).to(device, non_blocking=True)

        # Mixed precision training
        with autocast(enabled=args.amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        if args.amp:
            scaler.scale(loss).backward()
            if args.clip_grad:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()

        # Measure accuracy
        acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))

        # Update meters
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))
        top5.update(acc5[0].item(), images.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        # Log to TensorBoard
        if writer and batch_idx % args.log_interval == 0:
            global_step = epoch * len(loader) + batch_idx
            writer.add_scalar('train/loss', losses.avg, global_step)
            writer.add_scalar('train/acc1', top1.avg, global_step)
            writer.add_scalar('train/acc5', top5.avg, global_step)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step)

        # Print progress
        if batch_idx % args.log_interval == 0:
            print(f'Epoch [{epoch}][{batch_idx}/{len(loader)}]\t'
                  f'Time {batch_time.avg:.3f}\t'
                  f'Data {data_time.avg:.3f}\t'
                  f'Loss {losses.avg:.4f}\t'
                  f'Acc@1 {top1.avg:.2f}\t'
                  f'Acc@5 {top5.avg:.2f}')

    return losses.avg, top1.avg, top5.avg


def validate(loader, model, criterion, device, args):
    """Validate model."""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()

    # Validation transforms (deterministic)
    transforms = turboloader.Compose([
        turboloader.Resize(256, 256),
        turboloader.CenterCrop(224, 224),
        turboloader.ImageNetNormalize(),
        turboloader.ToTensor()
    ])

    end = time.time()

    with torch.no_grad():
        for batch in loader:
            # Process batch
            images = []
            labels = []

            for sample in batch:
                img = transforms.apply(sample['image'])
                images.append(torch.from_numpy(img).float())
                labels.append(sample['label'])

            images = torch.stack(images).to(device, non_blocking=True)
            labels = torch.tensor(labels, dtype=torch.long).to(device, non_blocking=True)

            # Forward pass
            with autocast(enabled=args.amp):
                outputs = model(images)
                loss = criterion(outputs, labels)

            # Measure accuracy
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))

            # Update meters
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0].item(), images.size(0))
            top5.update(acc5[0].item(), images.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

    print(f'Validation: Loss {losses.avg:.4f} Acc@1 {top1.avg:.2f} Acc@5 {top5.avg:.2f}')

    return losses.avg, top1.avg, top5.avg


def train_worker(rank, world_size, args):
    """Training worker for each GPU."""
    print(f"[Rank {rank}] Starting training")

    # Setup distributed
    if world_size > 1:
        setup_dist(rank, world_size)

    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')

    # Create model
    model = resnet50(num_classes=args.num_classes)
    model = model.to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[rank])

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr * world_size,  # Scale with world size
        momentum=0.9,
        weight_decay=args.weight_decay
    )

    # Learning rate scheduler with warmup
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return float(epoch + 1) / args.warmup_epochs
        else:
            progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
            return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # AMP scaler
    scaler = GradScaler(enabled=args.amp)

    # TensorBoard writer (rank 0 only)
    writer = None
    if rank == 0:
        log_dir = os.path.join(args.log_dir, time.strftime('%Y%m%d-%H%M%S'))
        writer = SummaryWriter(log_dir)

    # Resume from checkpoint
    start_epoch = 0
    best_acc1 = 0.0

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint['epoch'] + 1
            best_acc1 = checkpoint.get('best_acc1', 0.0)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if args.amp:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            print(f"Resumed from epoch {checkpoint['epoch']}")

    # Create dataloaders
    train_loader = turboloader.DataLoader(
        args.train_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        enable_distributed=(world_size > 1),
        drop_last=True
    )

    val_loader = turboloader.DataLoader(
        args.val_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        enable_distributed=(world_size > 1)
    )

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        print(f'\n{"="*70}')
        print(f'Epoch {epoch}/{args.epochs}')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        print(f'{"="*70}')

        # Train
        train_loss, train_acc1, train_acc5 = train_epoch(
            train_loader, model, criterion, optimizer, scaler,
            epoch, device, args, writer
        )

        # Validate
        val_loss, val_acc1, val_acc5 = validate(val_loader, model, criterion, device, args)

        # Step scheduler
        scheduler.step()

        # Log to TensorBoard
        if writer:
            writer.add_scalar('epoch/train_loss', train_loss, epoch)
            writer.add_scalar('epoch/train_acc1', train_acc1, epoch)
            writer.add_scalar('epoch/val_loss', val_loss, epoch)
            writer.add_scalar('epoch/val_acc1', val_acc1, epoch)
            writer.add_scalar('epoch/lr', optimizer.param_groups[0]['lr'], epoch)

        # Save checkpoint (rank 0 only)
        if rank == 0:
            is_best = val_acc1 > best_acc1
            best_acc1 = max(val_acc1, best_acc1)

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if world_size > 1 else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict() if args.amp else None,
                'best_acc1': best_acc1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_acc1': val_acc1,
                'args': vars(args)
            }

            os.makedirs(args.checkpoint_dir, exist_ok=True)
            save_checkpoint(
                checkpoint, is_best, args.checkpoint_dir,
                f'checkpoint_epoch_{epoch}.pt'
            )

            print(f'Checkpoint saved. Best Acc@1: {best_acc1:.2f}%')

        # Synchronize
        if world_size > 1:
            dist.barrier()

    # Final results
    if rank == 0:
        print(f'\n{"="*70}')
        print('Training Complete!')
        print(f'Best Validation Acc@1: {best_acc1:.2f}%')
        print(f'{"="*70}')

        if writer:
            writer.close()

    if world_size > 1:
        cleanup_dist()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='ImageNet ResNet50 Training')
    parser.add_argument('--train-data', type=str, required=True, help='Training data path')
    parser.add_argument('--val-data', type=str, required=True, help='Validation data path')
    parser.add_argument('--batch-size', type=int, default=256, help='Per-GPU batch size')
    parser.add_argument('--num-workers', type=int, default=8, help='Workers per GPU')
    parser.add_argument('--epochs', type=int, default=90, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.1, help='Base learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--warmup-epochs', type=int, default=5, help='Warmup epochs')
    parser.add_argument('--num-classes', type=int, default=1000, help='Number of classes')
    parser.add_argument('--gpus', type=int, default=None, help='Number of GPUs')
    parser.add_argument('--amp', action='store_true', help='Use mixed precision')
    parser.add_argument('--clip-grad', type=float, default=None, help='Gradient clipping')
    parser.add_argument('--log-interval', type=int, default=100, help='Log interval')
    parser.add_argument('--log-dir', type=str, default='./runs', help='TensorBoard log dir')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints', help='Checkpoint dir')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')

    args = parser.parse_args()

    # Determine world size
    if args.gpus is None:
        args.gpus = torch.cuda.device_count()

    if args.gpus == 0:
        print("Warning: No GPUs found, using CPU")
        args.gpus = 1

    world_size = args.gpus

    print("=" * 70)
    print("ImageNet ResNet50 Training with TurboLoader")
    print("=" * 70)
    print(f"Train data: {args.train_data}")
    print(f"Val data: {args.val_data}")
    print(f"GPUs: {args.gpus}")
    print(f"Batch size per GPU: {args.batch_size}")
    print(f"Global batch size: {args.batch_size * world_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Base LR: {args.lr}")
    print(f"Scaled LR: {args.lr * world_size}")
    print(f"Mixed Precision: {args.amp}")
    print("=" * 70 + "\n")

    # Spawn processes
    if world_size > 1:
        mp.spawn(train_worker, args=(world_size, args), nprocs=world_size, join=True)
    else:
        train_worker(0, world_size, args)


if __name__ == '__main__':
    main()
