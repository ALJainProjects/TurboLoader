#!/usr/bin/env python3
"""
Full ImageNet to WebDataset TAR Converter

Converts full ImageNet ILSVRC2012 dataset to WebDataset TAR format for TurboLoader.

Requirements:
1. Download ImageNet from image-net.org (requires account)
2. Extract to /path/to/imagenet/train/ (1000 class folders)
3. Run this script to convert to WebDataset TAR format

Usage:
    python3 benchmarks/imagenet_converter.py \
        --imagenet-dir /path/to/imagenet/train \
        --output-tar /path/to/imagenet_train.tar \
        --num-samples 1000000  # Optional: limit for testing
"""

import argparse
import tarfile
import json
import os
import io
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import hashlib

def convert_imagenet_to_tar(imagenet_dir, output_tar, num_samples=None, shard_size=10000):
    """
    Convert ImageNet directory structure to WebDataset TAR format.

    Args:
        imagenet_dir: Path to ImageNet train directory (contains n* class folders)
        output_tar: Output TAR file path (or base path for sharded TARs)
        num_samples: Optional limit on number of samples (for testing)
        shard_size: Number of samples per shard (default 10K for easier handling)
    """
    imagenet_dir = Path(imagenet_dir)

    if not imagenet_dir.exists():
        raise ValueError(f"ImageNet directory not found: {imagenet_dir}")

    # Get all class directories (n01440764, n01443537, etc.)
    class_dirs = sorted([d for d in imagenet_dir.iterdir() if d.is_dir() and d.name.startswith('n')])

    if not class_dirs:
        raise ValueError(f"No class directories found in {imagenet_dir}")

    print(f"Found {len(class_dirs)} classes in {imagenet_dir}")

    # Build class_name -> class_index mapping
    class_to_idx = {d.name: idx for idx, d in enumerate(class_dirs)}

    # Collect all image paths
    print("Collecting image paths...")
    all_samples = []
    for class_dir in tqdm(class_dirs, desc="Scanning classes"):
        class_name = class_dir.name
        class_idx = class_to_idx[class_name]

        # Find all JPEG images
        for img_path in class_dir.glob("*.JPEG"):
            all_samples.append({
                'path': img_path,
                'class_name': class_name,
                'class_idx': class_idx,
                'filename': img_path.name
            })

    print(f"Found {len(all_samples)} total images")

    # Limit samples if requested
    if num_samples and num_samples < len(all_samples):
        print(f"Limiting to {num_samples} samples")
        all_samples = all_samples[:num_samples]

    # Determine if we need sharding
    use_sharding = len(all_samples) > shard_size

    if use_sharding:
        num_shards = (len(all_samples) + shard_size - 1) // shard_size
        print(f"Creating {num_shards} shards of ~{shard_size} samples each")

        # Create output directory for shards
        output_path = Path(output_tar)
        shard_dir = output_path.parent / f"{output_path.stem}_shards"
        shard_dir.mkdir(exist_ok=True, parents=True)

        shard_paths = []
        for shard_idx in range(num_shards):
            start_idx = shard_idx * shard_size
            end_idx = min(start_idx + shard_size, len(all_samples))
            shard_samples = all_samples[start_idx:end_idx]

            shard_path = shard_dir / f"imagenet_train_{shard_idx:04d}.tar"
            print(f"\nCreating shard {shard_idx+1}/{num_shards}: {shard_path}")
            _write_tar(shard_path, shard_samples, start_idx)
            shard_paths.append(shard_path)

        # Write shard list
        shard_list_path = shard_dir / "shard_list.txt"
        with open(shard_list_path, 'w') as f:
            for path in shard_paths:
                f.write(f"{path}\n")

        print(f"\n✅ Created {num_shards} shards in {shard_dir}")
        print(f"   Shard list: {shard_list_path}")
        return shard_paths

    else:
        # Single TAR file
        print(f"Creating single TAR file: {output_tar}")
        _write_tar(output_tar, all_samples, 0)
        print(f"✅ Created {output_tar}")
        return [output_tar]


def _write_tar(tar_path, samples, global_offset=0):
    """Write samples to a TAR file in WebDataset format"""

    with tarfile.open(tar_path, 'w') as tar:
        for i, sample in enumerate(tqdm(samples, desc=f"Writing {Path(tar_path).name}")):
            sample_id = global_offset + i
            basename = f"sample_{sample_id:08d}"

            try:
                # Read and validate image
                img = Image.open(sample['path'])
                img = img.convert('RGB')  # Ensure RGB

                # Convert to JPEG bytes
                img_bytes = io.BytesIO()
                img.save(img_bytes, format='JPEG', quality=95)
                img_data = img_bytes.getvalue()

                # Write image to TAR
                img_info = tarfile.TarInfo(name=f"{basename}.jpg")
                img_info.size = len(img_data)
                tar.addfile(img_info, io.BytesIO(img_data))

                # Create metadata
                metadata = {
                    'class_name': sample['class_name'],
                    'class_idx': sample['class_idx'],
                    'filename': sample['filename'],
                    'width': img.width,
                    'height': img.height,
                }

                metadata_json = json.dumps(metadata).encode('utf-8')
                meta_info = tarfile.TarInfo(name=f"{basename}.json")
                meta_info.size = len(metadata_json)
                tar.addfile(meta_info, io.BytesIO(metadata_json))

            except Exception as e:
                print(f"\n⚠️  Error processing {sample['path']}: {e}")
                continue


def verify_tar(tar_path, num_samples_check=10):
    """Verify TAR file integrity by reading a few samples"""
    print(f"\nVerifying {tar_path}...")

    with tarfile.open(tar_path, 'r') as tar:
        members = tar.getmembers()
        print(f"  Total files in TAR: {len(members)}")

        # Check a few random samples
        jpg_files = [m for m in members if m.name.endswith('.jpg')]
        json_files = [m for m in members if m.name.endswith('.json')]

        print(f"  Images: {len(jpg_files)}")
        print(f"  Metadata files: {len(json_files)}")

        if len(jpg_files) != len(json_files):
            print(f"  ⚠️  Warning: Mismatch between images and metadata")

        # Read a few samples
        print(f"\n  Checking {num_samples_check} random samples...")
        for i, (jpg_member, json_member) in enumerate(zip(jpg_files[:num_samples_check],
                                                           json_files[:num_samples_check])):
            # Read image
            img_file = tar.extractfile(jpg_member)
            img_data = img_file.read()
            img = Image.open(io.BytesIO(img_data))

            # Read metadata
            meta_file = tar.extractfile(json_member)
            metadata = json.load(meta_file)

            print(f"    Sample {i}: {img.size} {img.mode} - class {metadata['class_name']}")

    print(f"✅ Verification complete\n")


def main():
    parser = argparse.ArgumentParser(description='Convert ImageNet to WebDataset TAR')
    parser.add_argument('--imagenet-dir', type=str, required=True,
                       help='Path to ImageNet train directory')
    parser.add_argument('--output-tar', type=str, required=True,
                       help='Output TAR file path (or base path for shards)')
    parser.add_argument('--num-samples', type=int, default=None,
                       help='Limit number of samples (for testing)')
    parser.add_argument('--shard-size', type=int, default=10000,
                       help='Number of samples per shard (default: 10000)')
    parser.add_argument('--verify', action='store_true',
                       help='Verify TAR files after creation')

    args = parser.parse_args()

    print("=" * 80)
    print("ImageNet → WebDataset TAR Converter")
    print("=" * 80)
    print(f"Input directory: {args.imagenet_dir}")
    print(f"Output TAR: {args.output_tar}")
    if args.num_samples:
        print(f"Sample limit: {args.num_samples}")
    print(f"Shard size: {args.shard_size}")
    print("=" * 80)

    # Convert
    tar_paths = convert_imagenet_to_tar(
        args.imagenet_dir,
        args.output_tar,
        num_samples=args.num_samples,
        shard_size=args.shard_size
    )

    # Verify
    if args.verify:
        for tar_path in tar_paths[:3]:  # Verify first 3 shards
            verify_tar(tar_path)

    print("\n" + "=" * 80)
    print("✅ CONVERSION COMPLETE")
    print("=" * 80)
    print(f"Output files: {len(tar_paths)}")

    if len(tar_paths) == 1:
        print(f"  {tar_paths[0]}")
    else:
        shard_dir = Path(tar_paths[0]).parent
        print(f"  Shard directory: {shard_dir}")
        print(f"  Shard list: {shard_dir / 'shard_list.txt'}")

    print("\nNext steps:")
    print(f"  python3 benchmarks/imagenet_benchmark.py --tar-paths {tar_paths[0]}")
    print("=" * 80)


if __name__ == '__main__':
    main()
