#!/usr/bin/env python3
"""
ImageNet Dataset Downloader

Downloads and prepares the ImageNet ILSVRC2012 dataset for benchmarking.

ImageNet ILSVRC2012:
- Training set: ~1.28M images, 1000 classes
- Validation set: 50K images, 1000 classes
- Image size: Variable (typically resized to 224x224 or 256x256)
- Total size: ~155 GB (training + validation)

NOTE: ImageNet requires registration and manual download from the official website.
This script provides helper functions to organize the dataset after download.

Official website: https://image-net.org/challenges/LSVRC/2012/

Usage:
    1. Register at https://image-net.org/
    2. Download ILSVRC2012_img_train.tar and ILSVRC2012_img_val.tar
    3. Run this script to extract and organize the dataset
"""

import os
import sys
import tarfile
import argparse
import shutil
from pathlib import Path
from typing import Optional
import json


def extract_train_set(train_tar: str, output_dir: str, progress: bool = True):
    """
    Extract ImageNet training set.

    The training TAR contains class-specific TAR files (n01440764.tar, etc.)
    This extracts them all and organizes into class directories.

    Args:
        train_tar: Path to ILSVRC2012_img_train.tar
        output_dir: Output directory for extracted images
        progress: Show progress
    """
    print("="*80)
    print("EXTRACTING IMAGENET TRAINING SET")
    print("="*80)
    print(f"Source: {train_tar}")
    print(f"Output: {output_dir}")
    print("="*80)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Extract main TAR
    print("\nExtracting main TAR file...")
    with tarfile.open(train_tar, 'r') as tar:
        members = tar.getmembers()
        print(f"Found {len(members)} class TAR files")

        for i, member in enumerate(members):
            if progress and (i + 1) % 100 == 0:
                print(f"  Extracted {i + 1}/{len(members)} class TARs...")

            # Extract class TAR to temp location
            tar.extract(member, output_path)

            # Class name from TAR file (e.g., n01440764.tar -> n01440764)
            class_tar_path = output_path / member.name
            class_name = Path(member.name).stem

            # Create class directory
            class_dir = output_path / class_name
            class_dir.mkdir(exist_ok=True)

            # Extract images from class TAR
            with tarfile.open(class_tar_path, 'r') as class_tar:
                class_tar.extractall(class_dir)

            # Remove class TAR
            class_tar_path.unlink()

    print(f"\nExtraction complete!")
    print(f"Training images organized in: {output_dir}")


def extract_val_set(val_tar: str, output_dir: str, labels_file: Optional[str] = None):
    """
    Extract ImageNet validation set.

    The validation TAR contains flat images. Optionally organize by class
    if labels file is provided.

    Args:
        val_tar: Path to ILSVRC2012_img_val.tar
        output_dir: Output directory for extracted images
        labels_file: Optional path to validation labels file
    """
    print("="*80)
    print("EXTRACTING IMAGENET VALIDATION SET")
    print("="*80)
    print(f"Source: {val_tar}")
    print(f"Output: {output_dir}")
    print("="*80)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\nExtracting validation TAR...")
    with tarfile.open(val_tar, 'r') as tar:
        tar.extractall(output_path)

    image_files = list(output_path.glob('*.JPEG'))
    print(f"Extracted {len(image_files)} validation images")

    # If labels file provided, organize by class
    if labels_file and os.path.exists(labels_file):
        print(f"\nOrganizing by class using: {labels_file}")
        organize_val_by_class(output_path, labels_file)

    print(f"\nExtraction complete!")
    print(f"Validation images in: {output_dir}")


def organize_val_by_class(val_dir: Path, labels_file: str):
    """
    Organize validation images into class directories.

    Args:
        val_dir: Directory containing validation images
        labels_file: File with validation labels
    """
    # Read labels
    with open(labels_file, 'r') as f:
        labels = f.read().strip().split('\n')

    image_files = sorted(val_dir.glob('*.JPEG'))

    if len(labels) != len(image_files):
        print(f"Warning: Label count ({len(labels)}) doesn't match image count ({len(image_files)})")
        return

    # Create class directories and move images
    for img_file, label in zip(image_files, labels):
        class_dir = val_dir / label
        class_dir.mkdir(exist_ok=True)
        shutil.move(str(img_file), str(class_dir / img_file.name))

    print(f"Organized {len(image_files)} images into {len(set(labels))} classes")


def create_tar_archive(input_dir: str, output_tar: str):
    """
    Create TAR archive from extracted ImageNet dataset.

    Args:
        input_dir: Directory with organized images
        output_tar: Output TAR file path
    """
    print("="*80)
    print("CREATING TAR ARCHIVE")
    print("="*80)
    print(f"Source: {input_dir}")
    print(f"Output: {output_tar}")
    print("="*80)

    input_path = Path(input_dir)

    # Count images
    image_files = list(input_path.rglob('*.JPEG'))
    print(f"\nFound {len(image_files)} images")

    print("\nCreating TAR archive...")
    with tarfile.open(output_tar, 'w') as tar:
        for i, img_file in enumerate(image_files):
            if (i + 1) % 10000 == 0:
                print(f"  Archived {i + 1}/{len(image_files)} images...")

            # Store with relative path
            arcname = img_file.relative_to(input_path)
            tar.add(img_file, arcname=str(arcname))

    tar_size_mb = os.path.getsize(output_tar) / (1024 * 1024)
    print(f"\nTAR archive created: {output_tar}")
    print(f"Size: {tar_size_mb:.2f} MB")


def create_metadata(dataset_dir: str, output_file: str):
    """
    Create metadata file for ImageNet dataset.

    Args:
        dataset_dir: Directory containing organized dataset
        output_file: Output JSON file
    """
    dataset_path = Path(dataset_dir)

    # Count classes and images
    class_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
    class_names = sorted([d.name for d in class_dirs])

    metadata = {
        'dataset': 'ImageNet ILSVRC2012',
        'num_classes': len(class_names),
        'classes': class_names,
        'class_counts': {}
    }

    for class_dir in class_dirs:
        image_count = len(list(class_dir.glob('*.JPEG')))
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
        description='ImageNet dataset downloader and organizer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract training set
  python download_imagenet.py --train-tar ILSVRC2012_img_train.tar --output imagenet/train

  # Extract validation set
  python download_imagenet.py --val-tar ILSVRC2012_img_val.tar --output imagenet/val

  # Create TAR archive for benchmarking
  python download_imagenet.py --create-tar imagenet/train --output imagenet_train.tar

  # Create metadata file
  python download_imagenet.py --create-metadata imagenet/train --output metadata.json
        """
    )

    parser.add_argument('--train-tar', type=str,
                       help='Path to ILSVRC2012_img_train.tar')
    parser.add_argument('--val-tar', type=str,
                       help='Path to ILSVRC2012_img_val.tar')
    parser.add_argument('--val-labels', type=str,
                       help='Path to validation labels file')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Output directory or file')
    parser.add_argument('--create-tar', type=str,
                       help='Create TAR archive from directory')
    parser.add_argument('--create-metadata', type=str,
                       help='Create metadata JSON from directory')

    args = parser.parse_args()

    # Extract training set
    if args.train_tar:
        if not os.path.exists(args.train_tar):
            print(f"Error: Training TAR not found: {args.train_tar}")
            sys.exit(1)
        extract_train_set(args.train_tar, args.output)

    # Extract validation set
    elif args.val_tar:
        if not os.path.exists(args.val_tar):
            print(f"Error: Validation TAR not found: {args.val_tar}")
            sys.exit(1)
        extract_val_set(args.val_tar, args.output, args.val_labels)

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
        create_metadata(args.create_metadata, args.output)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
