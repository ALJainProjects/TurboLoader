#!/usr/bin/env python3.13
"""
Simple benchmark comparing TurboLoader vs PyTorch DataLoader
"""
import sys
import os
import time
from pathlib import Path

# Add build directory to path for TurboLoader
sys.path.insert(0, str(Path(__file__).parent / "build"))

print("=" * 80)
print("TURBOLOADER vs PYTORCH DATALOADER - BENCHMARK")
print("=" * 80)
print()

# Configuration
TAR_PATH = "/tmp/benchmark_1000_webdataset.tar"
BATCH_SIZE = 32
NUM_EPOCHS = 3

print(f"Dataset: {TAR_PATH}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Number of epochs: {NUM_EPOCHS}")
print()

# ============================================================================
# TurboLoader Benchmark
# ============================================================================

print("=" * 80)
print("TURBOLOADER BENCHMARK")
print("=" * 80)

try:
    import turboloader

    for num_workers in [1, 2, 4, 8]:
        print(f"\n[TurboLoader - {num_workers} workers]")

        pipeline = turboloader.Pipeline(
            tar_paths=[TAR_PATH],
            num_workers=num_workers,
            decode_jpeg=True,
            queue_size=128,
            shuffle=False
        )

        samples_processed = 0
        start_time = time.time()

        for epoch in range(NUM_EPOCHS):
            if epoch > 0:
                pipeline.reset()

            pipeline.start()

            while True:
                batch = pipeline.next_batch(BATCH_SIZE)
                if len(batch) == 0:
                    break
                samples_processed += len(batch)

            pipeline.stop()

        elapsed = time.time() - start_time
        throughput = samples_processed / elapsed

        print(f"  Samples: {samples_processed}")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Throughput: {throughput:.2f} samples/sec")

except Exception as e:
    print(f"TurboLoader error: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 80)
print("BENCHMARK COMPLETE")
print("=" * 80)
