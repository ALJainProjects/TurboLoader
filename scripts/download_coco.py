#!/usr/bin/env python3
"""
COCO Dataset Downloader

Downloads and prepares the COCO (Common Objects in Context) dataset for benchmarking.

COCO 2017:
- Training set: 118K images
- Validation set: 5K images
- Test set: 41K images
- Annotations: Instance segmentation, keypoints, captions, panoptic
- Total size: ~25 GB (images) + annotations

Official website: https://cocodataset.org/

Usage:
    python download_coco.py --split train --output coco/train
    python download_coco.py --split val --output coco/val
"""

import os
import sys
import json
import argparse
import tarfile
import zipfile
import shutil
from pathlib import Path
from typing import Optional
import urllib.request


# COCO 2017 download URLs
COCO_URLS = {
    'train_images': 'http://images.cocodataset.org/zips/train2017.zip',
    'val_images': 'http://images.cocodataset.org/zips/val2017.zip',
    'test_images': 'http://images.cocodataset.org/zips/test2017.zip',
    'annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
}


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


def extract_zip(zip_path: str, output_dir: str):
    """
    Extract ZIP archive.

    Args:
        zip_path: Path to ZIP file
        output_dir: Output directory
    """
    print(f"Extracting: {zip_path}")
    print(f"  Output: {output_dir}")

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)

    print(f"  Extracted {len(zip_ref.namelist())} files")


def download_and_extract(split: str, output_dir: str, keep_zip: bool = False):
    """
    Download and extract COCO dataset split.

    Args:
        split: Dataset split ('train', 'val', or 'test')
        output_dir: Output directory
        keep_zip: Keep downloaded ZIP files
    """
    print("="*80)
    print(f"DOWNLOADING COCO {split.upper()} SET")
    print("="*80)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Download images
    if split == 'train':
        url = COCO_URLS['train_images']
        zip_name = 'train2017.zip'
    elif split == 'val':
        url = COCO_URLS['val_images']
        zip_name = 'val2017.zip'
    elif split == 'test':
        url = COCO_URLS['test_images']
        zip_name = 'test2017.zip'
    else:
        raise ValueError(f"Invalid split: {split}")

    zip_path = output_path / zip_name

    # Download
    if not zip_path.exists():
        download_file(url, str(zip_path))
    else:
        print(f"Using existing ZIP: {zip_path}")

    # Extract
    extract_zip(str(zip_path), str(output_path))

    # Download annotations (for train and val)
    if split in ['train', 'val']:
        ann_zip = output_path / 'annotations_trainval2017.zip'
        if not ann_zip.exists():
            print("\nDownloading annotations...")
            download_file(COCO_URLS['annotations'], str(ann_zip))
        else:
            print(f"\nUsing existing annotations: {ann_zip}")

        extract_zip(str(ann_zip), str(output_path))

        # Clean up annotation ZIP
        if not keep_zip:
            ann_zip.unlink()

    # Clean up image ZIP
    if not keep_zip:
        zip_path.unlink()

    print(f"\nDataset extracted to: {output_dir}")


def create_tar_archive(input_dir: str, output_tar: str):
    """
    Create TAR archive from COCO images.

    Args:
        input_dir: Directory with COCO images
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
    image_files = sorted(input_path.glob('**/*.jpg'))
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


def create_metadata(dataset_dir: str, annotations_file: Optional[str], output_file: str):
    """
    Create metadata file for COCO dataset.

    Args:
        dataset_dir: Directory containing COCO images
        annotations_file: Path to COCO annotations JSON (optional)
        output_file: Output JSON file
    """
    dataset_path = Path(dataset_dir)

    # Count images
    image_files = list(dataset_path.rglob('*.jpg'))

    metadata = {
        'dataset': 'COCO',
        'total_images': len(image_files),
        'image_dir': str(dataset_dir)
    }

    # Load annotations if provided
    if annotations_file and os.path.exists(annotations_file):
        with open(annotations_file, 'r') as f:
            coco_data = json.load(f)

        metadata['num_categories'] = len(coco_data.get('categories', []))
        metadata['num_annotations'] = len(coco_data.get('annotations', []))
        metadata['categories'] = [cat['name'] for cat in coco_data.get('categories', [])]

    # Save metadata
    with open(output_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nMetadata saved to: {output_file}")
    print(f"Total images: {len(image_files)}")
    if annotations_file:
        print(f"Categories: {metadata.get('num_categories', 0)}")
        print(f"Annotations: {metadata.get('num_annotations', 0)}")


def main():
    parser = argparse.ArgumentParser(
        description='COCO dataset downloader and organizer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download training set
  python download_coco.py --split train --output coco/train

  # Download validation set
  python download_coco.py --split val --output coco/val

  # Create TAR archive for benchmarking
  python download_coco.py --create-tar coco/train/train2017 --output coco_train.tar

  # Create metadata file
  python download_coco.py --create-metadata coco/train/train2017 \\
      --annotations coco/train/annotations/instances_train2017.json \\
      --output metadata.json
        """
    )

    parser.add_argument('--split', type=str, choices=['train', 'val', 'test'],
                       help='Dataset split to download')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Output directory or file')
    parser.add_argument('--keep-zip', action='store_true',
                       help='Keep downloaded ZIP files')
    parser.add_argument('--create-tar', type=str,
                       help='Create TAR archive from directory')
    parser.add_argument('--create-metadata', type=str,
                       help='Create metadata JSON from directory')
    parser.add_argument('--annotations', type=str,
                       help='Path to COCO annotations JSON file')

    args = parser.parse_args()

    # Download and extract
    if args.split:
        download_and_extract(args.split, args.output, args.keep_zip)

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
        create_metadata(args.create_metadata, args.annotations, args.output)

    else:
        parser.print_help()
        sys.exit(1)

    print("\n" + "="*80)
    print("DONE!")
    print("="*80)


if __name__ == '__main__':
    main()
