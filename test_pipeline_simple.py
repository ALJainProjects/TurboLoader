#!/usr/bin/env python3
"""
Simple TurboLoader Pipeline test to diagnose issues
"""
import turboloader
import time

print("TurboLoader version:", turboloader.__version__)
print("Creating pipeline...")

pipeline = turboloader.Pipeline(
    tar_paths=["/tmp/benchmark_1k.tar"],
    num_workers=2,
    queue_size=64,
    decode_jpeg=True
)

print(f"Total samples: {pipeline.total_samples()}")
print("Starting pipeline...")
pipeline.start()

print("Waiting 2 seconds for pipeline to fill queue...")
time.sleep(2)

print("\nTrying to fetch first batch...")
batch = pipeline.next_batch(8)
print(f"Got {len(batch)} samples in first batch")

if len(batch) > 0:
    sample = batch[0]
    print(f"First sample: index={sample.index}, shape=({sample.height}, {sample.width}, {sample.channels})")

    print("\nFetching 10 more batches...")
    for i in range(10):
        batch = pipeline.next_batch(8)
        print(f"  Batch {i+1}: {len(batch)} samples")
        if len(batch) == 0:
            print("  Pipeline exhausted!")
            break
else:
    print("ERROR: First batch was empty!")
    print("This indicates a problem with the pipeline implementation.")

pipeline.stop()
print("\nDone!")
