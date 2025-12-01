#!/usr/bin/env python3
"""
Synthetic Dataset Generator for Benchmarking

Generates synthetic image datasets in various formats for benchmarking:
- Raw image files (JPEG/PNG)
- TAR archive
- TBL v2 format (if available)
"""

import argparse
import os
import sys
import tarfile
import tempfile
import struct
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

try:
    from PIL import Image

    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("Warning: Pillow not installed. Using raw numpy arrays.")


def generate_random_image(width: int, height: int, channels: int = 3) -> np.ndarray:
    """Generate a random image with some structure (not pure noise)"""
    # Create base image with gradients and patterns for more realistic compression
    img = np.zeros((height, width, channels), dtype=np.uint8)

    # Add gradient background
    for c in range(channels):
        gradient = np.linspace(0, 255, width, dtype=np.uint8)
        img[:, :, c] = gradient

    # Add some random rectangles
    num_rects = np.random.randint(3, 10)
    for _ in range(num_rects):
        x1 = np.random.randint(0, width - 20)
        y1 = np.random.randint(0, height - 20)
        x2 = x1 + np.random.randint(10, min(100, width - x1))
        y2 = y1 + np.random.randint(10, min(100, height - y1))
        color = np.random.randint(0, 255, channels, dtype=np.uint8)
        img[y1:y2, x1:x2] = color

    # Add some noise
    noise = np.random.randint(-20, 20, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img


def save_image_jpeg(img: np.ndarray, path: str, quality: int = 85):
    """Save image as JPEG"""
    if HAS_PIL:
        Image.fromarray(img).save(path, "JPEG", quality=quality)
    else:
        # Fallback: save as raw binary
        with open(path, "wb") as f:
            f.write(img.tobytes())


def save_image_png(img: np.ndarray, path: str):
    """Save image as PNG"""
    if HAS_PIL:
        Image.fromarray(img).save(path, "PNG")
    else:
        with open(path, "wb") as f:
            f.write(img.tobytes())


def generate_dataset_files(
    output_dir: str,
    num_images: int,
    image_size: tuple = (256, 256),
    num_classes: int = 1000,
    format: str = "jpeg",
    num_workers: int = 4,
) -> list:
    """Generate dataset as individual image files"""
    os.makedirs(output_dir, exist_ok=True)

    files = []
    labels = []

    def generate_one(idx):
        img = generate_random_image(image_size[0], image_size[1])
        label = idx % num_classes

        if format == "jpeg":
            filename = f"img_{idx:08d}.jpg"
            filepath = os.path.join(output_dir, filename)
            save_image_jpeg(img, filepath)
        else:
            filename = f"img_{idx:08d}.png"
            filepath = os.path.join(output_dir, filename)
            save_image_png(img, filepath)

        return filepath, label

    print(f"Generating {num_images} images...")
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(generate_one, i) for i in range(num_images)]

        for i, future in enumerate(as_completed(futures)):
            filepath, label = future.result()
            files.append(filepath)
            labels.append(label)

            if (i + 1) % 1000 == 0:
                print(f"  Generated {i + 1}/{num_images} images")

    # Save labels file
    labels_path = os.path.join(output_dir, "labels.txt")
    with open(labels_path, "w") as f:
        for filepath, label in zip(files, labels):
            f.write(f"{os.path.basename(filepath)},{label}\n")

    print(f"Generated {len(files)} images in {output_dir}")
    return files


def generate_tar_dataset(
    output_path: str,
    num_images: int,
    image_size: tuple = (256, 256),
    num_classes: int = 1000,
    format: str = "jpeg",
) -> str:
    """Generate dataset as a TAR archive"""
    import io

    print(f"Generating TAR archive with {num_images} images...")

    with tarfile.open(output_path, "w") as tar:
        labels = []

        for idx in range(num_images):
            img = generate_random_image(image_size[0], image_size[1])
            label = idx % num_classes

            # Create in-memory file
            if HAS_PIL:
                buf = io.BytesIO()
                if format == "jpeg":
                    Image.fromarray(img).save(buf, "JPEG", quality=85)
                    filename = f"img_{idx:08d}.jpg"
                else:
                    Image.fromarray(img).save(buf, "PNG")
                    filename = f"img_{idx:08d}.png"
                buf.seek(0)
                data = buf.getvalue()
            else:
                data = img.tobytes()
                filename = f"img_{idx:08d}.raw"

            # Add to TAR
            info = tarfile.TarInfo(name=filename)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

            labels.append((filename, label))

            if (idx + 1) % 1000 == 0:
                print(f"  Added {idx + 1}/{num_images} images to TAR")

        # Add labels file
        labels_content = "\n".join(f"{f},{l}" for f, l in labels)
        labels_data = labels_content.encode("utf-8")
        info = tarfile.TarInfo(name="labels.txt")
        info.size = len(labels_data)
        tar.addfile(info, io.BytesIO(labels_data))

    print(f"Generated TAR archive: {output_path}")
    return output_path


def generate_varying_size_dataset(
    output_path: str,
    num_images: int,
    min_size: int = 128,
    max_size: int = 512,
    num_classes: int = 1000,
) -> str:
    """Generate dataset with varying image sizes (for smart batching benchmarks)"""
    import io

    print(f"Generating varying-size TAR archive with {num_images} images...")

    sizes = []
    with tarfile.open(output_path, "w") as tar:
        labels = []

        for idx in range(num_images):
            # Random size
            width = np.random.randint(min_size, max_size + 1)
            height = np.random.randint(min_size, max_size + 1)
            sizes.append((width, height))

            img = generate_random_image(width, height)
            label = idx % num_classes

            if HAS_PIL:
                buf = io.BytesIO()
                Image.fromarray(img).save(buf, "JPEG", quality=85)
                filename = f"img_{idx:08d}.jpg"
                buf.seek(0)
                data = buf.getvalue()
            else:
                data = img.tobytes()
                filename = f"img_{idx:08d}.raw"

            info = tarfile.TarInfo(name=filename)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

            labels.append((filename, label, width, height))

            if (idx + 1) % 1000 == 0:
                print(f"  Added {idx + 1}/{num_images} images")

        # Add metadata
        meta = {
            "num_images": num_images,
            "sizes": sizes,
            "labels": [(f, l) for f, l, w, h in labels],
        }
        meta_data = json.dumps(meta).encode("utf-8")
        info = tarfile.TarInfo(name="metadata.json")
        info.size = len(meta_data)
        tar.addfile(info, io.BytesIO(meta_data))

    print(f"Generated varying-size TAR: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic datasets for benchmarking")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output path (directory for files, .tar for archive)",
    )
    parser.add_argument(
        "--num-images", "-n", type=int, default=10000, help="Number of images to generate"
    )
    parser.add_argument("--width", type=int, default=256, help="Image width")
    parser.add_argument("--height", type=int, default=256, help="Image height")
    parser.add_argument(
        "--num-classes", type=int, default=1000, help="Number of classes for labels"
    )
    parser.add_argument("--format", choices=["jpeg", "png"], default="jpeg", help="Image format")
    parser.add_argument(
        "--type",
        choices=["files", "tar", "varying"],
        default="tar",
        help="Dataset type (files, tar archive, or varying sizes)",
    )
    parser.add_argument("--workers", type=int, default=4, help="Number of worker threads")

    args = parser.parse_args()

    if args.type == "files":
        generate_dataset_files(
            args.output,
            args.num_images,
            (args.width, args.height),
            args.num_classes,
            args.format,
            args.workers,
        )
    elif args.type == "tar":
        generate_tar_dataset(
            args.output, args.num_images, (args.width, args.height), args.num_classes, args.format
        )
    elif args.type == "varying":
        generate_varying_size_dataset(
            args.output, args.num_images, min_size=128, max_size=512, num_classes=args.num_classes
        )


if __name__ == "__main__":
    main()
