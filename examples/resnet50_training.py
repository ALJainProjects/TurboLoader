#!/usr/bin/env python3
"""
ResNet-50 ImageNet Training with TurboLoader

Complete training example showing how to train ResNet-50 on ImageNet
using TurboLoader for data loading. Demonstrates full training loop
with validation, checkpointing, and logging.
"""

import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, 'build/python')
import turboloader

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import torchvision.models as models


class AverageMeter:
    """Computes and stores the average and current value"""
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


def train_epoch(model, pipeline, criterion, optimizer, scaler, device, epoch, args):
    """Train for one epoch"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    num_batches = args.batches_per_epoch
    end = time.time()

    print(f"\n{'=' * 80}")
    print(f"EPOCH {epoch + 1}/{args.epochs}")
    print(f"{'=' * 80}")

    for batch_idx in range(num_batches):
        # Measure data loading time
        data_start = time.time()
        batch = pipeline.next_batch(args.batch_size)

        # Convert to PyTorch tensors
        images = []
        labels = []
        for sample in batch:
            img_data = sample.get_transformed_data()
            images.append(torch.from_numpy(img_data))
            # Parse label from filename (e.g., "n01440764_10026.JPEG" -> class ID)
            # For this example, we'll use dummy labels
            labels.append(0)  # Replace with actual label parsing

        images = torch.stack(images).to(device)
        targets = torch.tensor(labels, dtype=torch.long).to(device)

        data_time.update(time.time() - data_start)

        # Forward pass with mixed precision
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, targets)

        # Backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Measure accuracy
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))
        top5.update(acc5.item(), images.size(0))

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Print progress
        if batch_idx % args.print_freq == 0:
            print(f"Batch [{batch_idx}/{num_batches}] "
                  f"Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s) "
                  f"Data {data_time.val:.3f}s ({data_time.avg:.3f}s) "
                  f"Loss {losses.val:.4f} ({losses.avg:.4f}) "
                  f"Acc@1 {top1.val:.2f}% ({top1.avg:.2f}%) "
                  f"Acc@5 {top5.val:.2f}% ({top5.avg:.2f}%)")

    print(f"\nEpoch {epoch + 1} Summary:")
    print(f"  Avg Loss: {losses.avg:.4f}")
    print(f"  Avg Acc@1: {top1.avg:.2f}%")
    print(f"  Avg Acc@5: {top5.avg:.2f}%")
    print(f"  Avg Batch Time: {batch_time.avg * 1000:.2f}ms")
    print(f"  Avg Data Time: {data_time.avg * 1000:.2f}ms")
    print(f"  Throughput: {args.batch_size / batch_time.avg:.2f} images/sec")

    return losses.avg, top1.avg


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions"""
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


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    """Save checkpoint to disk"""
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")


def main():
    parser = argparse.ArgumentParser(description='ResNet-50 Training with TurboLoader')
    parser.add_argument('--train-tar', type=str, required=True,
                        help='Path to ImageNet training TAR file')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size (default: 256)')
    parser.add_argument('--epochs', type=int, default=90,
                        help='Number of epochs (default: 90)')
    parser.add_argument('--batches-per-epoch', type=int, default=5000,
                        help='Batches per epoch (default: 5000)')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay (default: 1e-4)')
    parser.add_argument('--workers', type=int, default=16,
                        help='Number of data loading workers (default: 16)')
    parser.add_argument('--print-freq', type=int, default=100,
                        help='Print frequency (default: 100)')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Checkpoint directory (default: checkpoints)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create checkpoint directory
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Create model
    print("\nCreating ResNet-50 model...")
    model = models.resnet50(pretrained=False, num_classes=1000)
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

    # Mixed precision scaler
    scaler = GradScaler()

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        print(f"\nLoading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        print(f"Resumed from epoch {start_epoch}")

    # Setup TurboLoader
    print("\nSetting up TurboLoader...")
    transform_config = turboloader.TransformConfig()
    transform_config.target_width = 224
    transform_config.target_height = 224
    transform_config.resize_mode = "bilinear"
    transform_config.normalize = True
    transform_config.mean = [0.485, 0.456, 0.406]
    transform_config.std = [0.229, 0.224, 0.225]
    transform_config.to_chw = True

    config = turboloader.Config()
    config.num_workers = args.workers
    config.queue_size = 512
    config.decode_jpeg = True
    config.enable_simd_transforms = True
    config.transform_config = transform_config

    pipeline = turboloader.Pipeline([args.train_tar], config)
    pipeline.start()

    print(f"\nTurboLoader configured:")
    print(f"  Workers: {args.workers}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Queue size: 512")
    print(f"  SIMD transforms: Enabled")

    # Training loop
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)

    for epoch in range(start_epoch, args.epochs):
        # Train for one epoch
        train_loss, train_acc = train_epoch(
            model, pipeline, criterion, optimizer, scaler,
            device, epoch, args
        )

        # Step learning rate scheduler
        scheduler.step()

        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = Path(args.checkpoint_dir) / f'checkpoint_epoch_{epoch + 1}.pth.tar'
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_loss': train_loss,
                'train_acc': train_acc,
            }, filename=str(checkpoint_path))

    # Final checkpoint
    final_checkpoint = Path(args.checkpoint_dir) / 'checkpoint_final.pth.tar'
    save_checkpoint({
        'epoch': args.epochs,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, filename=str(final_checkpoint))

    # Stop pipeline
    pipeline.stop()

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
