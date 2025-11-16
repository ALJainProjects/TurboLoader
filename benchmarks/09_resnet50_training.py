#!/usr/bin/env python3
"""
Full ML Training Pipeline - ResNet-50 on Synthetic Dataset

Demonstrates a complete training pipeline using TurboLoader:
- ResNet-50 architecture
- Cross-entropy loss
- SGD optimizer with momentum
- 5 training epochs
- Validation metrics

This benchmark shows TurboLoader in a realistic ML training scenario.
"""

import argparse
import json
import time
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import models
except ImportError:
    print("Error: PyTorch is required for this benchmark")
    print("Install with: pip install torch torchvision")
    sys.exit(1)

try:
    from _turboloader import DataLoader as TurboDataLoader
except ImportError:
    print("Error: TurboLoader Python bindings not found")
    print("Build with: cd build && cmake .. && make -j$(nproc) && cd .. && pip install -e .")
    sys.exit(1)


class ResNet50Trainer:
    """Trainer for ResNet-50 using TurboLoader"""

    def __init__(self, num_classes: int = 10, device: str = 'cpu'):
        """Initialize trainer with ResNet-50 model"""
        self.device = torch.device(device)

        # Load pretrained ResNet-50 (for architecture, random weights for fairness)
        self.model = models.resnet50(pretrained=False, num_classes=num_classes)
        self.model = self.model.to(self.device)

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer: SGD with momentum (standard for ResNet)
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=0.1,
            momentum=0.9,
            weight_decay=1e-4
        )

        # Learning rate scheduler: step decay
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=2,
            gamma=0.1
        )

        # Metrics
        self.train_losses: List[float] = []
        self.train_accuracies: List[float] = []
        self.epoch_times: List[float] = []

    def train_epoch(self, tar_path: str, batch_size: int,
                   num_workers: int, epoch: int) -> Dict:
        """Train for one epoch using TurboLoader"""
        self.model.train()

        # Create TurboLoader for this epoch
        loader = TurboDataLoader(
            data_path=tar_path,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True  # Shuffle for training
        )

        epoch_start = time.time()

        running_loss = 0.0
        correct = 0
        total = 0
        batch_count = 0

        # Training loop
        while not loader.is_finished():
            batch = loader.next_batch()
            if batch is None:
                break

            batch_count += 1

            # Convert to PyTorch tensors
            images = torch.from_numpy(batch['images']).float()
            labels = torch.from_numpy(batch['labels']).long()

            # Move to device
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Metrics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_time = time.time() - epoch_start

        # Calculate epoch metrics
        avg_loss = running_loss / batch_count if batch_count > 0 else 0.0
        accuracy = 100.0 * correct / total if total > 0 else 0.0

        self.train_losses.append(avg_loss)
        self.train_accuracies.append(accuracy)
        self.epoch_times.append(epoch_time)

        # Learning rate schedule step
        self.scheduler.step()

        return {
            'epoch': epoch,
            'loss': avg_loss,
            'accuracy': accuracy,
            'time': epoch_time,
            'batches': batch_count,
            'samples': total,
            'learning_rate': self.scheduler.get_last_lr()[0]
        }

    def get_summary(self) -> Dict:
        """Get training summary statistics"""
        return {
            'total_epochs': len(self.epoch_times),
            'total_time': sum(self.epoch_times),
            'avg_epoch_time': np.mean(self.epoch_times),
            'std_epoch_time': np.std(self.epoch_times),
            'final_loss': self.train_losses[-1] if self.train_losses else 0.0,
            'final_accuracy': self.train_accuracies[-1] if self.train_accuracies else 0.0,
            'best_accuracy': max(self.train_accuracies) if self.train_accuracies else 0.0,
            'epoch_times': self.epoch_times,
            'losses': self.train_losses,
            'accuracies': self.train_accuracies
        }


def run_training_benchmark(tar_path: str, batch_size: int = 32,
                          num_workers: int = 8, num_epochs: int = 5,
                          num_classes: int = 10, device: str = 'cpu') -> Dict:
    """Run full training benchmark with ResNet-50"""

    print("=" * 80)
    print("ResNet-50 Training Pipeline with TurboLoader")
    print("=" * 80)
    print(f"Dataset: {tar_path}")
    print(f"Batch size: {batch_size}")
    print(f"Workers: {num_workers}")
    print(f"Epochs: {num_epochs}")
    print(f"Device: {device}")
    print(f"Classes: {num_classes}")
    print("=" * 80)
    print()

    # Initialize trainer
    trainer = ResNet50Trainer(num_classes=num_classes, device=device)

    # Count model parameters
    total_params = sum(p.numel() for p in trainer.model.parameters())
    trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)

    print(f"Model: ResNet-50")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print()

    # Training loop
    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")
        print("-" * 40)

        epoch_metrics = trainer.train_epoch(
            tar_path, batch_size, num_workers, epoch
        )

        print(f"  Loss: {epoch_metrics['loss']:.4f}")
        print(f"  Accuracy: {epoch_metrics['accuracy']:.2f}%")
        print(f"  Time: {epoch_metrics['time']:.2f}s")
        print(f"  Batches: {epoch_metrics['batches']}")
        print(f"  Samples: {epoch_metrics['samples']}")
        print(f"  Learning rate: {epoch_metrics['learning_rate']:.6f}")
        print()

    # Get summary
    summary = trainer.get_summary()

    print("=" * 80)
    print("Training Summary")
    print("=" * 80)
    print(f"Total epochs: {summary['total_epochs']}")
    print(f"Total time: {summary['total_time']:.2f}s")
    print(f"Avg epoch time: {summary['avg_epoch_time']:.2f}s ± {summary['std_epoch_time']:.2f}s")
    print(f"Final loss: {summary['final_loss']:.4f}")
    print(f"Final accuracy: {summary['final_accuracy']:.2f}%")
    print(f"Best accuracy: {summary['best_accuracy']:.2f}%")
    print("=" * 80)

    # Build results dictionary
    results = {
        'framework': 'TurboLoader + PyTorch ResNet-50',
        'model': 'ResNet-50',
        'total_parameters': int(total_params),
        'trainable_parameters': int(trainable_params),
        'batch_size': batch_size,
        'num_workers': num_workers,
        'num_epochs': num_epochs,
        'num_classes': num_classes,
        'device': device,
        'optimizer': 'SGD(lr=0.1, momentum=0.9, weight_decay=1e-4)',
        'scheduler': 'StepLR(step_size=2, gamma=0.1)',
        'loss_function': 'CrossEntropyLoss',
        'total_time': summary['total_time'],
        'avg_epoch_time': summary['avg_epoch_time'],
        'std_epoch_time': summary['std_epoch_time'],
        'epoch_times': summary['epoch_times'],
        'final_loss': summary['final_loss'],
        'final_accuracy': summary['final_accuracy'],
        'best_accuracy': summary['best_accuracy'],
        'losses_per_epoch': summary['losses'],
        'accuracies_per_epoch': summary['accuracies']
    }

    return results


def main():
    parser = argparse.ArgumentParser(
        description='ResNet-50 training benchmark with TurboLoader',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('tar_path', help='Path to TAR dataset')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--num-workers', type=int, default=8,
                       help='Number of data loading workers')
    parser.add_argument('--num-epochs', type=int, default=5,
                       help='Number of training epochs')
    parser.add_argument('--num-classes', type=int, default=10,
                       help='Number of output classes')
    parser.add_argument('--device', default='cpu',
                       choices=['cpu', 'cuda', 'mps'],
                       help='Device to use for training')
    parser.add_argument('--output', type=str,
                       default='benchmark_results/09_resnet50_training.json',
                       help='Output JSON file for results')

    args = parser.parse_args()

    # Validate dataset exists
    if not Path(args.tar_path).exists():
        print(f"Error: Dataset not found at {args.tar_path}")
        sys.exit(1)

    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        args.device = 'cpu'
    elif args.device == 'mps' and not torch.backends.mps.is_available():
        print("Warning: MPS not available, falling back to CPU")
        args.device = 'cpu'

    # Run benchmark
    results = run_training_benchmark(
        args.tar_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_epochs=args.num_epochs,
        num_classes=args.num_classes,
        device=args.device
    )

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
