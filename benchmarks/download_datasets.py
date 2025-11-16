#!/usr/bin/env python3
"""
Dataset Downloader for TurboLoader Benchmarks

Downloads real-world datasets for comprehensive benchmarking:
- Images: ImageNet, CIFAR-10, COCO
- Text: WikiText-103, AG News, IMDB
- Audio: LibriSpeech, Common Voice
- Video: UCF-101
- Multi-modal: COCO Captions

All datasets are converted to WebDataset TAR format for TurboLoader.
"""

import os
import sys
import argparse
import tarfile
import json
import urllib.request
import zipfile
import gzip
import shutil
from pathlib import Path
from tqdm import tqdm
import hashlib

# ============================================================================
# Utilities
# ============================================================================

def download_file(url, destination, desc=None):
    """Download file with progress bar"""
    print(f"Downloading {desc or url}...")

    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=desc) as t:
        urllib.request.urlretrieve(url, filename=destination, reporthook=t.update_to)

def verify_checksum(filepath, expected_md5):
    """Verify file checksum"""
    md5 = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5.update(chunk)
    return md5.hexdigest() == expected_md5

def create_tar_dataset(output_path, samples, desc="Creating TAR"):
    """Create WebDataset TAR from samples"""
    print(f"Creating {output_path}...")

    with tarfile.open(output_path, 'w') as tar:
        for i, sample in enumerate(tqdm(samples, desc=desc)):
            # Each sample is a dict with 'data', 'label', 'metadata', etc.
            basename = f"sample_{i:08d}"

            # Write data file
            if 'data' in sample:
                data_path = f"/tmp/{basename}.{sample.get('ext', 'bin')}"
                with open(data_path, 'wb') as f:
                    f.write(sample['data'])
                tar.add(data_path, arcname=f"{basename}.{sample.get('ext', 'bin')}")
                os.remove(data_path)

            # Write metadata JSON
            metadata = {
                'label': sample.get('label', 0),
                'index': i,
            }
            if 'metadata' in sample:
                metadata.update(sample['metadata'])

            meta_path = f"/tmp/{basename}.json"
            with open(meta_path, 'w') as f:
                json.dump(metadata, f)
            tar.add(meta_path, arcname=f"{basename}.json")
            os.remove(meta_path)

    print(f"✅ Created {output_path} ({os.path.getsize(output_path) / 1024 / 1024:.1f} MB)")

# ============================================================================
# Image Datasets
# ============================================================================

def download_cifar10(output_dir):
    """Download CIFAR-10 dataset"""
    print("\n" + "="*70)
    print("CIFAR-10: 60,000 32x32 color images in 10 classes")
    print("="*70)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    tar_path = output_dir / "cifar-10-python.tar.gz"

    # Download
    if not tar_path.exists():
        download_file(url, tar_path, "CIFAR-10")

    # Extract
    extract_dir = output_dir / "cifar-10-batches-py"
    if not extract_dir.exists():
        print("Extracting...")
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(output_dir)

    # Convert to WebDataset format
    import pickle

    samples = []
    for batch_num in range(1, 6):
        batch_file = extract_dir / f"data_batch_{batch_num}"
        with open(batch_file, 'rb') as f:
            data = pickle.load(f, encoding='bytes')

            for i in range(len(data[b'labels'])):
                # CIFAR-10 images are 3x32x32 in RGB
                img_flat = data[b'data'][i]
                img_rgb = img_flat.reshape(3, 32, 32).transpose(1, 2, 0)

                # Convert to JPEG
                from PIL import Image
                import io
                img = Image.fromarray(img_rgb)
                img_bytes = io.BytesIO()
                img.save(img_bytes, format='JPEG', quality=95)

                samples.append({
                    'data': img_bytes.getvalue(),
                    'ext': 'jpg',
                    'label': int(data[b'labels'][i])
                })

    # Create TAR
    create_tar_dataset(
        output_dir / "cifar10_train.tar",
        samples,
        "Creating CIFAR-10 TAR"
    )

    return output_dir / "cifar10_train.tar"

def download_imagenet_sample(output_dir):
    """Download ImageNet sample (tiny-imagenet-200)"""
    print("\n" + "="*70)
    print("Tiny ImageNet: 200 classes, 100K images (64x64)")
    print("Full ImageNet requires manual download from image-net.org")
    print("="*70)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_path = output_dir / "tiny-imagenet-200.zip"

    # Download
    if not zip_path.exists():
        download_file(url, zip_path, "Tiny ImageNet")

    # Extract
    extract_dir = output_dir / "tiny-imagenet-200"
    if not extract_dir.exists():
        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)

    # Convert to WebDataset format
    from PIL import Image
    import io

    samples = []
    train_dir = extract_dir / "train"

    for class_dir in tqdm(list(train_dir.iterdir()), desc="Processing classes"):
        if not class_dir.is_dir():
            continue

        label = class_dir.name
        images_dir = class_dir / "images"

        for img_path in images_dir.glob("*.JPEG"):
            # Read and convert to JPEG bytes
            img = Image.open(img_path).convert('RGB')
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='JPEG', quality=95)

            samples.append({
                'data': img_bytes.getvalue(),
                'ext': 'jpg',
                'label': label,
                'metadata': {'class': label}
            })

    # Create TAR
    create_tar_dataset(
        output_dir / "tiny_imagenet_train.tar",
        samples,
        "Creating Tiny ImageNet TAR"
    )

    print("\n📝 Note: For full ImageNet, download from https://image-net.org/")
    print("   Then run: python convert_imagenet_to_tar.py /path/to/imagenet")

    return output_dir / "tiny_imagenet_train.tar"

# ============================================================================
# Text Datasets
# ============================================================================

def download_ag_news(output_dir):
    """Download AG News classification dataset"""
    print("\n" + "="*70)
    print("AG News: 120K news articles in 4 categories")
    print("="*70)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # AG News is available via HuggingFace datasets
    try:
        from datasets import load_dataset
        print("Loading AG News from HuggingFace...")
        dataset = load_dataset('ag_news', split='train')

        samples = []
        for item in tqdm(dataset, desc="Processing"):
            text = item['text']
            label = item['label']

            samples.append({
                'data': text.encode('utf-8'),
                'ext': 'txt',
                'label': label
            })

        # Create TAR
        create_tar_dataset(
            output_dir / "ag_news_train.tar",
            samples,
            "Creating AG News TAR"
        )

        return output_dir / "ag_news_train.tar"

    except ImportError:
        print("⚠️  Install datasets: pip install datasets")
        return None

def download_wikitext103(output_dir):
    """Download WikiText-103 dataset"""
    print("\n" + "="*70)
    print("WikiText-103: 100M+ tokens from Wikipedia")
    print("="*70)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    url = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip"
    zip_path = output_dir / "wikitext-103-raw-v1.zip"

    # Download
    if not zip_path.exists():
        download_file(url, zip_path, "WikiText-103")

    # Extract
    extract_dir = output_dir / "wikitext-103-raw"
    if not extract_dir.exists():
        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)

    # Read and chunk text
    train_file = extract_dir / "wiki.train.raw"

    samples = []
    chunk_size = 512  # Characters per sample

    print("Processing text...")
    with open(train_file, 'r', encoding='utf-8') as f:
        text = f.read()

        # Split into chunks
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i+chunk_size]
            if len(chunk) > 100:  # Skip very small chunks
                samples.append({
                    'data': chunk.encode('utf-8'),
                    'ext': 'txt',
                    'label': 0
                })

    # Create TAR
    create_tar_dataset(
        output_dir / "wikitext103_train.tar",
        samples[:100000],  # Limit to 100K samples for benchmark
        "Creating WikiText-103 TAR"
    )

    return output_dir / "wikitext103_train.tar"

# ============================================================================
# Audio Datasets
# ============================================================================

def download_librispeech_sample(output_dir):
    """Download LibriSpeech sample"""
    print("\n" + "="*70)
    print("LibriSpeech: Clean speech (100 hours sample)")
    print("="*70)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download dev-clean (small subset)
    url = "https://www.openslr.org/resources/12/dev-clean.tar.gz"
    tar_path = output_dir / "dev-clean.tar.gz"

    if not tar_path.exists():
        download_file(url, tar_path, "LibriSpeech dev-clean")

    # Extract
    extract_dir = output_dir / "LibriSpeech"
    if not extract_dir.exists():
        print("Extracting...")
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(output_dir)

    # Convert to WebDataset
    samples = []
    dev_clean = extract_dir / "dev-clean"

    for speaker_dir in tqdm(list(dev_clean.iterdir()), desc="Processing speakers"):
        if not speaker_dir.is_dir():
            continue

        for chapter_dir in speaker_dir.iterdir():
            if not chapter_dir.is_dir():
                continue

            for audio_file in chapter_dir.glob("*.flac"):
                # Read audio file
                with open(audio_file, 'rb') as f:
                    audio_data = f.read()

                # Read transcript
                trans_file = audio_file.with_suffix('.txt')
                transcript = ""
                if trans_file.exists():
                    transcript = trans_file.read_text()

                samples.append({
                    'data': audio_data,
                    'ext': 'flac',
                    'label': 0,
                    'metadata': {
                        'transcript': transcript,
                        'speaker': speaker_dir.name,
                        'duration': 0  # Would need audio library to get actual duration
                    }
                })

    # Create TAR
    create_tar_dataset(
        output_dir / "librispeech_dev_clean.tar",
        samples[:5000],  # Limit for benchmark
        "Creating LibriSpeech TAR"
    )

    return output_dir / "librispeech_dev_clean.tar"

# ============================================================================
# Multi-modal Datasets
# ============================================================================

def download_coco_captions_sample(output_dir):
    """Download COCO Captions sample"""
    print("\n" + "="*70)
    print("COCO Captions: Images with text descriptions")
    print("Downloading validation set (5K images)")
    print("="*70)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download images
    images_url = "http://images.cocodataset.org/zips/val2017.zip"
    images_zip = output_dir / "val2017.zip"

    if not images_zip.exists():
        download_file(images_url, images_zip, "COCO val2017 images")

    # Download annotations
    annot_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    annot_zip = output_dir / "annotations_trainval2017.zip"

    if not annot_zip.exists():
        download_file(annot_url, annot_zip, "COCO annotations")

    # Extract
    images_dir = output_dir / "val2017"
    if not images_dir.exists():
        print("Extracting images...")
        with zipfile.ZipFile(images_zip, 'r') as zip_ref:
            zip_ref.extractall(output_dir)

    annotations_dir = output_dir / "annotations"
    if not annotations_dir.exists():
        print("Extracting annotations...")
        with zipfile.ZipFile(annot_zip, 'r') as zip_ref:
            zip_ref.extractall(output_dir)

    # Load captions
    captions_file = annotations_dir / "captions_val2017.json"
    with open(captions_file, 'r') as f:
        captions_data = json.load(f)

    # Create image_id -> captions mapping
    image_captions = {}
    for annot in captions_data['annotations']:
        image_id = annot['image_id']
        if image_id not in image_captions:
            image_captions[image_id] = []
        image_captions[image_id].append(annot['caption'])

    # Create samples
    samples = []
    for img_info in tqdm(captions_data['images'][:1000], desc="Processing"):  # Limit to 1K
        image_id = img_info['id']
        filename = img_info['file_name']
        img_path = images_dir / filename

        if img_path.exists() and image_id in image_captions:
            # Read image
            with open(img_path, 'rb') as f:
                img_data = f.read()

            # Get first caption
            caption = image_captions[image_id][0]

            samples.append({
                'data': img_data,
                'ext': 'jpg',
                'label': 0,
                'metadata': {
                    'caption': caption,
                    'image_id': image_id,
                    'all_captions': image_captions[image_id]
                }
            })

    # Create TAR
    create_tar_dataset(
        output_dir / "coco_captions_val_1k.tar",
        samples,
        "Creating COCO Captions TAR"
    )

    return output_dir / "coco_captions_val_1k.tar"

# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Download datasets for TurboLoader benchmarks')
    parser.add_argument('--output-dir', default='./datasets', help='Output directory')
    parser.add_argument('--datasets', nargs='+',
                       choices=['all', 'cifar10', 'imagenet', 'ag_news', 'wikitext',
                               'librispeech', 'coco'],
                       default=['all'],
                       help='Datasets to download')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = args.datasets
    if 'all' in datasets:
        datasets = ['cifar10', 'imagenet', 'ag_news', 'wikitext', 'librispeech', 'coco']

    downloaded = {}

    # Download each dataset
    if 'cifar10' in datasets:
        try:
            path = download_cifar10(output_dir / 'cifar10')
            downloaded['CIFAR-10'] = path
        except Exception as e:
            print(f"❌ Failed to download CIFAR-10: {e}")

    if 'imagenet' in datasets:
        try:
            path = download_imagenet_sample(output_dir / 'imagenet')
            downloaded['Tiny ImageNet'] = path
        except Exception as e:
            print(f"❌ Failed to download ImageNet: {e}")

    if 'ag_news' in datasets:
        try:
            path = download_ag_news(output_dir / 'ag_news')
            if path:
                downloaded['AG News'] = path
        except Exception as e:
            print(f"❌ Failed to download AG News: {e}")

    if 'wikitext' in datasets:
        try:
            path = download_wikitext103(output_dir / 'wikitext')
            downloaded['WikiText-103'] = path
        except Exception as e:
            print(f"❌ Failed to download WikiText: {e}")

    if 'librispeech' in datasets:
        try:
            path = download_librispeech_sample(output_dir / 'librispeech')
            downloaded['LibriSpeech'] = path
        except Exception as e:
            print(f"❌ Failed to download LibriSpeech: {e}")

    if 'coco' in datasets:
        try:
            path = download_coco_captions_sample(output_dir / 'coco')
            downloaded['COCO Captions'] = path
        except Exception as e:
            print(f"❌ Failed to download COCO: {e}")

    # Summary
    print("\n" + "="*70)
    print("DOWNLOAD SUMMARY")
    print("="*70)
    for name, path in downloaded.items():
        if path:
            size_mb = os.path.getsize(path) / 1024 / 1024
            print(f"✅ {name:20s} {path} ({size_mb:.1f} MB)")

    print("\n🚀 Ready to run benchmarks!")
    print(f"   python benchmarks/comprehensive_benchmark.py --datasets-dir {output_dir}")

if __name__ == '__main__':
    main()
