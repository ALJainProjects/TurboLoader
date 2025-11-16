#!/usr/bin/env python3
"""Test TurboLoader Python bindings"""

import turboloader
import time

def test_basic():
    """Test basic pipeline functionality"""
    print("=" * 60)
    print("Testing TurboLoader Python Bindings")
    print("=" * 60)
    print()

    # Create pipeline
    tar_files = [
        "/tmp/turboloader_real_images/shard_0000.tar",
        "/tmp/turboloader_real_images/shard_0001.tar"
    ]

    print(f"Creating pipeline with {len(tar_files)} TAR files...")
    pipeline = turboloader.Pipeline(
        tar_paths=tar_files,
        num_workers=4,
        decode_jpeg=True  # Enable JPEG decoding
    )

    print(f"Total samples: {pipeline.total_samples()}")
    print()

    # Start pipeline
    print("Starting pipeline...")
    pipeline.start()

    # Get a single batch
    print("Getting first batch (32 samples)...")
    batch = pipeline.next_batch(32)
    print(f"Batch size: {len(batch)}")

    if batch:
        sample = batch[0]
        print(f"\nFirst sample: {sample}")
        print(f"  Index: {sample.index}")
        print(f"  Width: {sample.width}")
        print(f"  Height: {sample.height}")
        print(f"  Channels: {sample.channels}")

        # Get image as NumPy array
        img = sample.get_image()
        print(f"  Image shape: {img.shape}")
        print(f"  Image dtype: {img.dtype}")

    print()

    # Benchmark full throughput
    print("Benchmarking throughput...")
    start = time.time()
    total_samples = 0
    batch_count = 0

    while True:
        batch = pipeline.next_batch(32)
        if not batch:
            break

        total_samples += len(batch)
        batch_count += 1

        if batch_count % 10 == 0:
            print(f"\r  Processed: {total_samples} samples...", end="", flush=True)

    elapsed = time.time() - start

    print(f"\r  Processed: {total_samples} samples    ")
    print()
    print(f"Results:")
    print(f"  Total samples: {total_samples}")
    print(f"  Time: {elapsed:.3f}s")
    print(f"  Throughput: {total_samples / elapsed:.1f} samples/sec")

    pipeline.stop()
    print()
    print("Test passed!")

def test_iterator():
    """Test batch iteration"""
    print()
    print("=" * 60)
    print("Testing Batch Iteration")
    print("=" * 60)
    print()

    tar_files = ["/tmp/turboloader_real_images/shard_0000.tar"]

    pipeline = turboloader.Pipeline(
        tar_paths=tar_files,
        num_workers=2,
        decode_jpeg=True
    )

    pipeline.start()

    print("Iterating with batch retrieval (batch_size=16)...")
    total = 0
    while True:
        batch = pipeline.next_batch(16)
        if not batch:
            break
        total += len(batch)
        if total >= 100:  # Just test first 100 samples
            break

    print(f"  Processed {total} samples")

    pipeline.stop()
    print()
    print("Batch iteration test passed!")

if __name__ == "__main__":
    test_basic()
    test_iterator()
