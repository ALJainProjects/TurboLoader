#!/usr/bin/env python3
"""
PyTorch Lightning Integration Example with TurboLoader

This example demonstrates how to integrate TurboLoader with PyTorch Lightning
for scalable, production-ready ML training.

Features demonstrated:
- LightningDataModule with TurboLoader
- Custom DataLoader wrapping
- Distributed training support
- Multi-GPU training
- Automatic logging and checkpointing

Requirements:
    pip install pytorch-lightning turboloader

Usage:
    # Single GPU
    python pytorch_lightning_example.py --data-path /path/to/data.tar

    # Multi-GPU
    python pytorch_lightning_example.py --data-path /path/to/data.tar --gpus 4

    # Distributed
    python pytorch_lightning_example.py --data-path /path/to/data.tar --strategy ddp --gpus 8
"""

import argparse
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import IterableDataset

try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
    from pytorch_lightning.loggers import TensorBoardLogger
except ImportError:
    print("Error: PyTorch Lightning not installed")
    print("Install with: pip install pytorch-lightning")
    exit(1)

import turboloader


class TurboLoaderWrapper(IterableDataset):
    """
    Wrapper to make TurboLoader compatible with PyTorch Lightning.

    PyTorch Lightning expects a PyTorch DataLoader, so we wrap TurboLoader
    as an IterableDataset.
    """

    def __init__(
        self,
        data_path: str,
        batch_size: int = 32,
        num_workers: int = 4,
        shuffle: bool = False,
        enable_distributed: bool = False,
        transform=None
    ):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.enable_distributed = enable_distributed
        self.transform = transform

        # Create TurboLoader instance
        self.loader = None

    def _create_loader(self):
        """Create TurboLoader with current settings."""
        return turboloader.DataLoader(
            self.data_path,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
            enable_distributed=self.enable_distributed,
        )

    def __iter__(self):
        """Iterate over TurboLoader batches."""
        if self.loader is None:
            self.loader = self._create_loader()

        for batch in self.loader:
            # Convert TurboLoader batch to PyTorch tensors
            images = []
            labels = []

            for sample in batch:
                img = sample['image']

                # Apply transforms if provided
                if self.transform:
                    img = self.transform(img)

                # Convert to tensor (assuming image is NumPy array)
                img_tensor = torch.from_numpy(img).float()

                # Permute from HWC to CHW if needed
                if img_tensor.ndim == 3 and img_tensor.shape[2] in [1, 3]:
                    img_tensor = img_tensor.permute(2, 0, 1)

                images.append(img_tensor)
                labels.append(sample.get('label', 0))

            # Stack into batch
            images = torch.stack(images)
            labels = torch.tensor(labels, dtype=torch.long)

            yield images, labels


class ImageNetDataModule(pl.LightningDataModule):
    """
    Lightning DataModule using TurboLoader for ImageNet-style datasets.
    """

    def __init__(
        self,
        train_path: str,
        val_path: Optional[str] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        img_size: int = 224
    ):
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path or train_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size

        # Save hyperparameters
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None):
        """Setup datasets."""
        # Create transforms
        self.train_transform = turboloader.Compose([
            turboloader.Resize(256, 256),
            turboloader.RandomCrop(self.img_size, self.img_size),
            turboloader.RandomHorizontalFlip(0.5),
            turboloader.ColorJitter(0.2, 0.2, 0.2, 0.1),
            turboloader.ImageNetNormalize()
        ])

        self.val_transform = turboloader.Compose([
            turboloader.Resize(self.img_size, self.img_size),
            turboloader.ImageNetNormalize()
        ])

    def train_dataloader(self):
        """Create training DataLoader."""
        dataset = TurboLoaderWrapper(
            self.train_path,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            enable_distributed=True,  # Enable for multi-GPU
            transform=self.train_transform
        )

        # Wrap in PyTorch DataLoader (batch_size=None since batching is done by TurboLoader)
        return TorchDataLoader(
            dataset,
            batch_size=None,
            num_workers=0  # TurboLoader handles workers
        )

    def val_dataloader(self):
        """Create validation DataLoader."""
        dataset = TurboLoaderWrapper(
            self.val_path,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            enable_distributed=True,
            transform=self.val_transform
        )

        return TorchDataLoader(
            dataset,
            batch_size=None,
            num_workers=0
        )


class ResNetClassifier(pl.LightningModule):
    """
    Simple ResNet-18 classifier for demonstration.
    """

    def __init__(
        self,
        num_classes: int = 1000,
        learning_rate: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 1e-4
    ):
        super().__init__()
        self.save_hyperparameters()

        # Use pretrained ResNet-18
        from torchvision.models import resnet18
        self.model = resnet18(num_classes=num_classes)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """Training step."""
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)

        # Calculate accuracy
        acc = (outputs.argmax(dim=1) == labels).float().mean()

        # Log metrics
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_acc', acc, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)

        # Calculate accuracy
        acc = (outputs.argmax(dim=1) == labels).float().mean()

        # Log metrics
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        self.log('val_acc', acc, prog_bar=True, on_epoch=True)

        return loss

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.learning_rate,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay
        )

        # Cosine annealing scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch"
            }
        }


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='PyTorch Lightning + TurboLoader Example')
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to training data (TAR or TBL format)')
    parser.add_argument('--val-path', type=str, default=None,
                        help='Path to validation data (default: use training data)')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size per GPU (default: 256)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')
    parser.add_argument('--num-classes', type=int, default=1000,
                        help='Number of classes (default: 1000)')
    parser.add_argument('--img-size', type=int, default=224,
                        help='Input image size (default: 224)')
    parser.add_argument('--epochs', type=int, default=90,
                        help='Number of training epochs (default: 90)')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Learning rate (default: 0.1)')
    parser.add_argument('--gpus', type=int, default=1,
                        help='Number of GPUs (default: 1)')
    parser.add_argument('--strategy', type=str, default='auto',
                        help='Training strategy: auto, ddp, ddp_spawn (default: auto)')
    parser.add_argument('--precision', type=str, default='32',
                        help='Training precision: 32, 16, bf16 (default: 32)')
    parser.add_argument('--log-dir', type=str, default='./lightning_logs',
                        help='TensorBoard log directory')

    args = parser.parse_args()

    # Create DataModule
    data_module = ImageNetDataModule(
        train_path=args.data_path,
        val_path=args.val_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size
    )

    # Create model
    model = ResNetClassifier(
        num_classes=args.num_classes,
        learning_rate=args.lr
    )

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        mode='max',
        save_top_k=3,
        filename='resnet18-{epoch:02d}-{val_acc:.2f}'
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Setup logger
    logger = TensorBoardLogger(args.log_dir, name='turboloader_resnet18')

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='gpu' if args.gpus > 0 else 'cpu',
        devices=args.gpus if args.gpus > 0 else 1,
        strategy=args.strategy,
        precision=args.precision,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
        log_every_n_steps=50
    )

    # Train!
    print("=" * 70)
    print("Starting training with TurboLoader + PyTorch Lightning")
    print("=" * 70)
    print(f"Data path: {args.data_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"GPUs: {args.gpus}")
    print(f"Strategy: {args.strategy}")
    print(f"Precision: {args.precision}")
    print("=" * 70)

    trainer.fit(model, data_module)

    print("\n" + "=" * 70)
    print("Training complete!")
    print(f"Best model saved to: {checkpoint_callback.best_model_path}")
    print(f"TensorBoard logs: {args.log_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
