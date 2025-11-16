#!/usr/bin/env python3
"""
CIFAR-10/100 Dataset Downloader

Downloads and prepares the CIFAR-10 and CIFAR-100 datasets for benchmarking.

CIFAR-10:
- Training set: 50K images, 10 classes
- Test set: 10K images
- Image size: 32x32 RGB
- Total size: ~170 MB

CIFAR-100:
- Training set: 50K images, 100 classes
- Test set: 10K images
- Image size: 32x32 RGB
- Total size: ~170 MB

Official website: https://www.cs.toronto.edu/~kriz/cifar.html

Usage:
    python download_cifar.py --dataset cifar10 --output cifar10/
    python download_cifar.py --dataset cifar100 --output cifar100/
"""

import os
import sys
import json
import argparse
import tarfile
import pickle
from pathlib import Path
from typing import List, Tuple, Dict
import urllib.request
import numpy as np
from PIL import Image


# CIFAR download URLs
CIFAR_URLS = {
    'cifar10': 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
    'cifar100': 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
}

# CIFAR-10 class names
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


def download_file(url: str, output_path: str, show_progress: bool = True):
    """
    Download file with progress bar.

    Args:
        url: URL to download
        output_path: Output file path
        show_progress: Show download progress
    """
    def reporthook(count, block_size, total_size):
        if not show_progress:
            return
        percent = min(int(count * block_size * 100 / total_size), 100)
        mb_downloaded = (count * block_size) / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        print(f"\r  Progress: {percent}% ({mb_downloaded:.1f}/{total_mb:.1f} MB)", end='')

    print(f"Downloading: {url}")
    urllib.request.urlretrieve(url, output_path, reporthook)
    print()  # New line after progress


def unpickle(file_path: str) -> dict:
    """
    Load CIFAR batch file.

    Args:
        file_path: Path to batch file

    Returns:
        Dictionary with batch data
    """
    with open(file_path, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    return data


def save_cifar10_images(batch_file: str, output_dir: str, class_names: List[str]):
    """
    Save CIFAR-10 images from batch file.

    Args:
        batch_file: Path to CIFAR-10 batch file
        output_dir: Output directory
        class_names: List of class names
    """
    output_path = Path(output_dir)

    # Load batch
    batch = unpickle(batch_file)
    images = batch[b'data']
    labels = batch[b'labels']
    filenames = batch[b'filenames']

    # Reshape images (3072 -> 32x32x3)
    images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    # Save images
    for img_data, label, filename in zip(images, labels, filenames):
        # Create class directory
        class_name = class_names[label]
        class_dir = output_path / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        # Save image
        if isinstance(filename, bytes):
            filename = filename.decode('utf-8')
        img = Image.fromarray(img_data)
        img.save(class_dir / filename)


def save_cifar100_images(batch_file: str, output_dir: str, meta_file: str):
    """
    Save CIFAR-100 images from batch file.

    Args:
        batch_file: Path to CIFAR-100 batch file
        output_dir: Output directory
        meta_file: Path to metadata file
    """
    output_path = Path(output_dir)

    # Load metadata for class names
    meta = unpickle(meta_file)
    fine_label_names = [name.decode('utf-8') if isinstance(name, bytes) else name
                       for name in meta[b'fine_label_names']]

    # Load batch
    batch = unpickle(batch_file)
    images = batch[b'data']
    fine_labels = batch[b'fine_labels']
    filenames = batch[b'filenames']

    # Reshape images
    images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    # Save images
    for img_data, label, filename in zip(images, fine_labels, filenames):
        # Create class directory
        class_name = fine_label_names[label]
        class_dir = output_path / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        # Save image
        if isinstance(filename, bytes):
            filename = filename.decode('utf-8')
        img = Image.fromarray(img_data)
        img.save(class_dir / filename)


def download_and_extract_cifar10(output_dir: str):
    """
    Download and extract CIFAR-10 dataset.

    Args:
        output_dir: Output directory
    """
    print("="*80)
    print("DOWNLOADING CIFAR-10 DATASET")
    print("="*80)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Download
    tar_path = output_path / 'cifar-10-python.tar.gz'
    if not tar_path.exists():
        download_file(CIFAR_URLS['cifar10'], str(tar_path))
    else:
        print(f"Using existing file: {tar_path}")

    # Extract TAR
    print("\nExtracting archive...")
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(output_path)

    cifar_dir = output_path / 'cifar-10-batches-py'

    # Save training images
    print("\nSaving training images...")
    train_dir = output_path / 'train'
    for i in range(1, 6):
        batch_file = cifar_dir / f'data_batch_{i}'
        print(f"  Processing batch {i}/5...")
        save_cifar10_images(str(batch_file), str(train_dir), CIFAR10_CLASSES)

    # Save test images
    print("\nSaving test images...")
    test_dir = output_path / 'test'
    test_batch = cifar_dir / 'test_batch'
    save_cifar10_images(str(test_batch), str(test_dir), CIFAR10_CLASSES)

    print(f"\nCIFAR-10 dataset saved to: {output_dir}")
    print(f"  Training images: {output_dir}/train/")
    print(f"  Test images: {output_dir}/test/")


def download_and_extract_cifar100(output_dir: str):
    """
    Download and extract CIFAR-100 dataset.

    Args:
        output_dir: Output directory
    """
    print("="*80)
    print("DOWNLOADING CIFAR-100 DATASET")
    print("="*80)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Download
    tar_path = output_path / 'cifar-100-python.tar.gz'
    if not tar_path.exists():
        download_file(CIFAR_URLS['cifar100'], str(tar_path))
    else:
        print(f"Using existing file: {tar_path}")

    # Extract TAR
    print("\nExtracting archive...")
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(output_path)

    cifar_dir = output_path / 'cifar-100-python'
    meta_file = cifar_dir / 'meta'

    # Save training images
    print("\nSaving training images...")
    train_dir = output_path / 'train'
    train_batch = cifar_dir / 'train'
    save_cifar100_images(str(train_batch), str(train_dir), str(meta_file))

    # Save test images
    print("\nSaving test images...")
    test_dir = output_path / 'test'
    test_batch = cifar_dir / 'test'
    save_cifar100_images(str(test_batch), str(test_dir), str(meta_file))

    print(f"\nCIFAR-100 dataset saved to: {output_dir}")
    print(f"  Training images: {output_dir}/train/")
    print(f"  Test images: {output_dir}/test/")


def create_tar_archive(input_dir: str, output_tar: str):
    """
    Create TAR archive from CIFAR images.

    Args:
        input_dir: Directory with CIFAR images
        output_tar: Output TAR file path
    """
    print("="*80)
    print("CREATING TAR ARCHIVE")
    print("="*80)
    print(f"Source: {input_dir}")
    print(f"Output: {output_tar}")
    print("="*80)

    input_path = Path(input_dir)

    # Find all images
    image_files = sorted(input_path.rglob('*.png'))
    print(f"\nFound {len(image_files)} images")

    print("\nCreating TAR archive...")
    with tarfile.open(output_tar, 'w') as tar:
        for i, img_file in enumerate(image_files):
            if (i + 1) % 5000 == 0:
                print(f"  Archived {i + 1}/{len(image_files)} images...")

            # Store with relative path
            arcname = img_file.relative_to(input_path)
            tar.add(img_file, arcname=str(arcname))

    tar_size_mb = os.path.getsize(output_tar) / (1024 * 1024)
    print(f"\nTAR archive created: {output_tar}")
    print(f"Size: {tar_size_mb:.2f} MB")


def create_metadata(dataset_dir: str, dataset_name: str, output_file: str):
    """
    Create metadata file for CIFAR dataset.

    Args:
        dataset_dir: Directory containing CIFAR images
        dataset_name: Dataset name ('cifar10' or 'cifar100')
        output_file: Output JSON file
    """
    dataset_path = Path(dataset_dir)

    # Count classes and images
    class_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
    class_names = sorted([d.name for d in class_dirs])

    metadata = {
        'dataset': f'CIFAR-{10 if dataset_name == "cifar10" else 100}',
        'image_size': '32x32',
        'num_classes': len(class_names),
        'classes': class_names,
        'class_counts': {}
    }

    for class_dir in class_dirs:
        image_count = len(list(class_dir.glob('*.png')))
        metadata['class_counts'][class_dir.name] = image_count

    total_images = sum(metadata['class_counts'].values())
    metadata['total_images'] = total_images

    # Save metadata
    with open(output_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nMetadata saved to: {output_file}")
    print(f"Total classes: {len(class_names)}")
    print(f"Total images: {total_images}")


def main():
    parser = argparse.ArgumentParser(
        description='CIFAR-10/100 dataset downloader',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download CIFAR-10
  python download_cifar.py --dataset cifar10 --output cifar10/

  # Download CIFAR-100
  python download_cifar.py --dataset cifar100 --output cifar100/

  # Create TAR archive for benchmarking
  python download_cifar.py --create-tar cifar10/train --output cifar10_train.tar

  # Create metadata file
  python download_cifar.py --create-metadata cifar10/train --dataset-name cifar10 --output metadata.json
        """
    )

    parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100'],
                       help='Dataset to download')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Output directory or file')
    parser.add_argument('--create-tar', type=str,
                       help='Create TAR archive from directory')
    parser.add_argument('--create-metadata', type=str,
                       help='Create metadata JSON from directory')
    parser.add_argument('--dataset-name', type=str, choices=['cifar10', 'cifar100'],
                       help='Dataset name for metadata')

    args = parser.parse_args()

    # Download and extract
    if args.dataset:
        if args.dataset == 'cifar10':
            download_and_extract_cifar10(args.output)
        else:
            download_and_extract_cifar100(args.output)

    # Create TAR archive
    elif args.create_tar:
        if not os.path.exists(args.create_tar):
            print(f"Error: Input directory not found: {args.create_tar}")
            sys.exit(1)
        create_tar_archive(args.create_tar, args.output)

    # Create metadata
    elif args.create_metadata:
        if not os.path.exists(args.create_metadata):
            print(f"Error: Dataset directory not found: {args.create_metadata}")
            sys.exit(1)
        if not args.dataset_name:
            print("Error: --dataset-name required for metadata creation")
            sys.exit(1)
        create_metadata(args.create_metadata, args.dataset_name, args.output)

    else:
        parser.print_help()
        sys.exit(1)

    print("\n" + "="*80)
    print("DONE!")
    print("="*80)


if __name__ == '__main__':
    main()
