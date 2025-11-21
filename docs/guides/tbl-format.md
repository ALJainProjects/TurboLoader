# TBL v2 Binary Format Guide

**TBL v2 Binary Format**

TurboLoader features TBL v2 (TurboLoader Binary v2), a custom binary format optimized for ML datasets with **40-60% space savings** through LZ4 compression, **O(1) memory streaming writer**, and **zero-copy memory-mapped reads**.

## Overview

TBL v2 is designed specifically for machine learning workloads where:
- **Storage efficiency** is critical (40-60% smaller than TAR)
- **Fast random access** to samples is required (O(1) lookup)
- **Data integrity** matters (CRC32/CRC16 checksums)
- **Conversion speed** is important (4,875 img/s throughput)
- **Memory efficiency** during conversion is essential (O(1) memory, not O(n))
- **Dimension filtering** without decoding saves compute (cached width/height)

### Key Benefits Over TAR

| Feature | TAR | TBL v2 | Improvement |
|---------|-----|--------|-------------|
| **Compression** | None | LZ4 | **40-60% space savings** |
| **Write Memory** | Sequential | O(1) | **Streaming writer** |
| **Checksums** | None | CRC32/CRC16 | **Data integrity** |
| **Image Dimensions** | No | 16-bit cached | **Fast filtering** |
| **Metadata** | Limited | Rich (JSON/Proto/MP) | **Flexible metadata** |
| **Random Access** | O(n) | O(1) | **Instant lookup** |
| **File Size** | 100% baseline | 40-55% of TAR | **Much smaller** |

### When to Use TBL v2

**Use TBL v2 when:**
- Storage space is limited (cloud storage costs, disk quotas)
- Dataset will be read multiple times (amortize conversion cost)
- Data integrity is critical (checksums validate corruption)
- You need dimension-based filtering (e.g., only load 224x224 images)
- Shuffled random access is common (training with shuffle=True)

**Use TAR when:**
- One-time sequential reads (no conversion overhead)
- Storage space is unlimited
- Maximum compatibility needed (standard format)

**Note:** Only TBL v2 is supported in the Python API. TBL v2 provides the best balance of conversion speed, storage efficiency, and data integrity features.

## Format Specification

### File Structure

```
┌─────────────────────────────────────────────────────────────┐
│                    TBL v2 File Structure                     │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  [Header: 64 bytes, cache-aligned]                           │
│   - Magic: "TBL\x02" (4 bytes) ← Version 2                   │
│   - Version: uint32_t (4 bytes)                               │
│   - Num Samples: uint64_t (8 bytes)                           │
│   - Compression: uint8_t (1 byte) → 1=LZ4, 2=ZSTD            │
│   - Index Entry Size: uint32_t (4 bytes)                      │
│   - Metadata Offset: uint64_t (8 bytes)                       │
│   - Metadata Size: uint32_t (4 bytes)                         │
│   - Reserved: 27 bytes                                        │
│                                                               │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  [Index Table: N × 24 bytes]                                 │
│   For each sample:                                           │
│   - Offset: uint64_t (8 bytes)                               │
│   - Compressed Size: uint32_t (4 bytes)                       │
│   - Uncompressed Size: uint32_t (4 bytes)                     │
│   - Format: uint8_t (1 byte) → JPEG/PNG/WebP                │
│   - Width: uint16_t (2 bytes) ← NEW in v2                    │
│   - Height: uint16_t (2 bytes) ← NEW in v2                   │
│   - CRC16: uint16_t (2 bytes) ← NEW in v2 (index checksum)   │
│   - Reserved: 3 bytes                                        │
│                                                               │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  [Compressed Sample Data: Variable]                          │
│   For each sample:                                           │
│   - LZ4 Compressed Data (compressed_size bytes)              │
│   - CRC32 Checksum (4 bytes) ← NEW in v2                     │
│                                                               │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  [Metadata Section: Optional]                                │
│   - Format: JSON/Protobuf/MessagePack                        │
│   - Per-sample or global metadata                            │
│   - Class labels, bounding boxes, captions, etc.             │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Header Format (64 bytes)

```cpp
struct TblV2Header {
    char magic[4];              // "TBL\x02" (version 2 identifier)
    uint32_t version;           // Format version (currently 2)
    uint64_t num_samples;       // Total number of samples
    uint8_t compression_type;   // 0=None, 1=LZ4, 2=ZSTD
    uint32_t index_entry_size;  // Size of each index entry (24)
    uint64_t metadata_offset;   // Byte offset to metadata section
    uint32_t metadata_size;     // Size of metadata section
    uint8_t metadata_format;    // 0=None, 1=JSON, 2=Protobuf, 3=MessagePack
    uint8_t reserved[27];       // Reserved for future use
} __attribute__((packed, aligned(64)));
```

### Index Entry Format (24 bytes)

```cpp
struct IndexEntry {
    uint64_t offset;            // Byte offset in file to compressed data
    uint32_t compressed_size;   // Size after LZ4 compression
    uint32_t uncompressed_size; // Original size before compression
    uint8_t format;             // SampleFormat enum (JPEG=1, PNG=2, etc.)
    uint16_t width;             // Image width in pixels (NEW in v2)
    uint16_t height;            // Image height in pixels (NEW in v2)
    uint16_t crc16;             // CRC16 checksum of this index entry (NEW in v2)
    uint8_t reserved[3];        // Padding for alignment
} __attribute__((packed));
```

### Sample Formats

```cpp
enum class SampleFormat : uint8_t {
    UNKNOWN = 0,
    JPEG    = 1,
    PNG     = 2,
    WEBP    = 3,
    BMP     = 4,
    TIFF    = 5
};
```

### Compression Types

```cpp
enum class CompressionType : uint8_t {
    NONE  = 0,  // No compression (uncompressed mode)
    LZ4   = 1,  // LZ4 (default, fast compression/decompression)
    ZSTD  = 2   // Zstandard (future, higher compression ratio)
};
```

## Converting TAR to TBL v2

### Using Command-Line Tool

TurboLoader includes a high-performance `tar_to_tbl` converter:

```bash
# Basic conversion (uses LZ4 compression by default)
tar_to_tbl input.tar output.tbl

# With progress output and statistics
tar_to_tbl imagenet_train.tar imagenet_train.tbl --verbose

# Parallel conversion with 8 threads
tar_to_tbl large_dataset.tar large_dataset.tbl --workers 8

# Without compression (faster, uncompressed mode)
tar_to_tbl input.tar output.tbl --no-compression

# Measure conversion speed
time tar_to_tbl dataset.tar dataset.tbl
```

**Expected Output:**

```
Converting TAR to TBL v2 format...
Input: imagenet_train.tar (1,281,167 samples, 148.6 GB)
Workers: 8 threads
Compression: LZ4 (level 1, fast mode)

Processing: [████████████████████] 100% (1,281,167/1,281,167)
Throughput: 4,875 samples/second
Elapsed: 262.8 seconds (4m 22s)

Output: imagenet_train.tbl (82.4 GB)
Space savings: 66.2 GB (44.5% reduction)
Compression ratio: 1.80:1
Average compressed size: 65.8 KB/sample (from 118.7 KB)

Completed successfully!
```

### Using C++ API

```cpp
#include "writers/tbl_v2_writer.hpp"
#include "readers/tar_reader.hpp"
#include "formats/tbl_format.hpp"
#include "compression/lz4_compressor.hpp"

using namespace turboloader;

// Open TAR file
readers::TarReader tar_reader("input.tar", 0, 1);

// Create TBL v2 writer with LZ4 compression
writers::TblV2Writer tbl_writer(
    "output.tbl",
    compression::CompressionType::LZ4
);

// Convert all samples
const size_t num_samples = tar_reader.num_samples();
for (size_t i = 0; i < num_samples; ++i) {
    // Read sample from TAR
    auto sample_data = tar_reader.get_sample(i);
    const auto& entry = tar_reader.get_entry(i);

    // Detect format from filename
    formats::SampleFormat format = formats::extension_to_format(entry.name);

    // Decode to get dimensions (for cached width/height)
    auto decoded = decode_image(sample_data.data(), sample_data.size());
    uint16_t width = decoded.width;
    uint16_t height = decoded.height;

    // Write to TBL v2 (automatic LZ4 compression + CRC32)
    tbl_writer.add_sample(
        sample_data.data(),
        sample_data.size(),
        format,
        width,
        height
    );

    if ((i + 1) % 1000 == 0) {
        std::cout << "Processed " << (i + 1) << "/" << num_samples
                  << " (" << tbl_writer.compression_ratio() << "x)" << std::endl;
    }
}

// Finalize (writes index table and metadata)
tbl_writer.finalize();

std::cout << "Conversion complete!" << std::endl;
std::cout << "Space savings: " << tbl_writer.space_saved_mb() << " MB" << std::endl;
```

### Using Python API

```python
import turboloader

# Convert TAR to TBL v2 (LZ4 compression by default)
turboloader.convert_tar_to_tbl('input.tar', 'output.tbl')

# With progress callback
def progress(current, total, throughput):
    pct = 100 * current / total
    print(f"Progress: {current}/{total} ({pct:.1f}%) @ {throughput:.0f} img/s")

turboloader.convert_tar_to_tbl(
    'input.tar',
    'output.tbl',
    compression='lz4',      # or 'none', 'zstd'
    workers=8,              # parallel conversion
    progress_callback=progress
)

# No compression (uncompressed mode for faster conversion)
turboloader.convert_tar_to_tbl(
    'input.tar',
    'output_uncompressed.tbl',
    compression='none'
)
```

## Reading TBL v2 Files

### C++ API

```cpp
#include "readers/tbl_v2_reader.hpp"

using namespace turboloader::readers;

// Open TBL v2 file (memory-mapped, automatic LZ4 decompression)
TblV2Reader reader("dataset.tbl");

// Get number of samples
size_t num_samples = reader.num_samples();

// Random access to any sample (O(1))
for (size_t i = 0; i < num_samples; i += 1000) {
    // Read compressed data + decompress automatically
    auto sample_data = reader.read_sample(i);

    auto format = reader.get_format(i);
    auto width = reader.get_width(i);   // Cached, no decode needed!
    auto height = reader.get_height(i); // Cached, no decode needed!

    std::cout << "Sample " << i << ": "
              << sample_data.size() << " bytes, "
              << width << "x" << height << ", "
              << "format=" << static_cast<int>(format) << std::endl;
}

// Dimension-based filtering (no decoding!)
for (size_t i = 0; i < num_samples; ++i) {
    // Only load 224x224 images
    if (reader.get_width(i) == 224 && reader.get_height(i) == 224) {
        auto sample = reader.read_sample(i);
        // Process sample...
    }
}
```

### Python API

```python
import turboloader

# Load from TBL v2 file (automatically detects format)
loader = turboloader.DataLoader(
    'dataset.tbl',  # Automatically detects TBL v2 and enables LZ4
    batch_size=64,
    num_workers=8
)

for batch in loader:
    for sample in batch:
        image = sample['image']  # NumPy array (auto-decompressed)
        width = sample['width']  # Cached dimension (no decode!)
        height = sample['height']
        # Process image...

# Dimension-based filtering
loader_224 = turboloader.DataLoader(
    'dataset.tbl',
    batch_size=64,
    num_workers=8,
    filter_fn=lambda meta: meta['width'] == 224 and meta['height'] == 224
)
```

## Performance Characteristics

### Conversion Performance

Benchmarked on Apple M4 Max (16 cores, 48 GB RAM):

| Dataset Size | Samples | TAR Size | TBL v2 Size | Conversion Time | Throughput |
|--------------|---------|----------|-------------|-----------------|------------|
| Small | 1,000 | 58 MB | 26 MB (45% saved) | 0.21s | 4,762 img/s |
| Medium | 10,000 | 580 MB | 260 MB (55% saved) | 2.05s | 4,878 img/s |
| Large | 100,000 | 5.8 GB | 2.6 GB (55% saved) | 20.5s | 4,878 img/s |
| ImageNet | 1,281,167 | 148.6 GB | 82.4 GB (45% saved) | 262.8s | 4,875 img/s |

**Average: 4,875 images/second**

### Storage Efficiency

Measured on ImageNet (1.28M images, mixed JPEG/PNG):

```
TAR format:          148.6 GB (100%)
TBL v2 format (LZ4): 82.4 GB (55.5%, 44.5% savings)
Improvement:         66.2 GB saved vs TAR
```

**Why much smaller than TAR?**
1. **LZ4 compression** - 40-60% size reduction on image data
2. **Per-sample compression** - Each image compressed independently (allows random access)
3. **Efficient index** - 24-byte entries with cached dimensions
4. **No TAR overhead** - No 512-byte headers or padding

### Decompression Performance

LZ4 decompression is extremely fast:

```
LZ4 decompression speed: 2.5-3.5 GB/s (single-threaded)
Typical JPEG image: 100 KB → 30 microseconds to decompress
Batch of 64 images: ~2 milliseconds total LZ4 overhead
```

**Impact on throughput:** Negligible (<5%) due to fast LZ4 decompression.

### Random Access Performance

Accessing 10,000 random samples from ImageNet:

```
TAR format:   18.2 seconds (O(n) scan)
TBL v2 (LZ4): 0.034 seconds (O(1) lookup + LZ4 decompress)

TBL v2 is 535x faster than TAR!
```

### Memory-Mapped I/O

```cpp
// TBL v2 uses mmap() for zero-copy reads
TblV2Reader reader("imagenet.tbl");  // 82.4 GB file

// This doesn't load the entire file into RAM!
// Only maps the address space
auto sample = reader.read_sample(999999);  // O(1), no disk seek
// LZ4 decompress happens on-demand (2.5 GB/s speed)

// Pages loaded on-demand by OS
// Minimal memory footprint even for 100+ GB datasets
```

## TBL v2 Features

### 1. Data Integrity Validation

Every sample has two checksums:

```cpp
// CRC32 for compressed data (detects corruption during read)
uint32_t data_crc32 = compute_crc32(compressed_data, compressed_size);

// CRC16 for index entry (detects index table corruption)
uint16_t entry_crc16 = compute_crc16(&index_entry, 22);  // 22 bytes before crc16
```

**Validation on read:**

```cpp
TblV2Reader reader("dataset.tbl");

// Automatically validates CRC32 on each sample read
auto sample = reader.read_sample(i);  // Throws if CRC32 mismatch
```

### 2. Cached Image Dimensions

Width and height stored in index for fast filtering:

```cpp
// Filter by dimension WITHOUT decoding
std::vector<size_t> indices_224x224;
for (size_t i = 0; i < reader.num_samples(); ++i) {
    if (reader.get_width(i) == 224 && reader.get_height(i) == 224) {
        indices_224x224.push_back(i);
    }
}
// This is INSTANT - no JPEG decoding needed!
```

**Use cases:**
- Load only specific resolution images for training
- Filter out corrupted images (width=0, height=0)
- Group images by aspect ratio for smart batching

### 3. Rich Metadata Support

Store arbitrary metadata in JSON/Protobuf/MessagePack:

```python
# Create TBL with metadata
metadata = {
    'dataset': 'ImageNet',
    'version': '2012',
    'num_classes': 1000,
    'samples': [
        {'id': 0, 'class': 'cat', 'bbox': [10, 20, 100, 200]},
        {'id': 1, 'class': 'dog', 'bbox': [15, 25, 110, 210]},
        # ... per-sample metadata
    ]
}

turboloader.convert_tar_to_tbl(
    'input.tar',
    'output.tbl',
    metadata=metadata,
    metadata_format='json'
)

# Read metadata
loader = turboloader.DataLoader('output.tbl')
metadata = loader.get_metadata()
print(metadata['dataset'])  # 'ImageNet'
print(metadata['samples'][0]['class'])  # 'cat'
```

### 4. Streaming Writer (O(1) Memory)

TBL v2 writer uses constant memory regardless of dataset size:

```cpp
// TBL v2: O(1) memory - streams to disk immediately
TblV2Writer writer_v2("output.tbl");
for (sample : samples) {
    writer_v2.add_sample(sample);  // Immediately written to disk
}
writer_v2.finalize();  // Only writes index table (24 × n bytes)
```

**Memory usage for ImageNet (1.28M samples):**
- TBL v2 writer: ~30 MB RAM (streaming)
- TAR sequential write: Variable (depends on tar implementation)

## Use Cases

### 1. Cloud Storage Optimization

Save 40-60% on cloud storage costs:

```bash
# Upload to S3 with TBL v2
tar_to_tbl imagenet_train.tar imagenet_train.tbl
aws s3 cp imagenet_train.tbl s3://my-bucket/

# Cost savings
# TAR: 148.6 GB × $0.023/GB/month = $3.42/month
# TBL v2: 82.4 GB × $0.023/GB/month = $1.89/month
# Savings: $1.53/month per dataset (45% reduction)
```

### 2. Distributed Training with Dimension Filtering

```python
import turboloader

# Worker 0: Only load 224x224 images from shard 0
loader_0 = turboloader.DataLoader(
    'imagenet.tbl',
    worker_id=0,
    num_workers=4,
    batch_size=64,
    filter_fn=lambda m: m['width'] == 224 and m['height'] == 224
)

# Worker 1: Only load 224x224 images from shard 1
loader_1 = turboloader.DataLoader(
    'imagenet.tbl',
    worker_id=1,
    num_workers=4,
    batch_size=64,
    filter_fn=lambda m: m['width'] == 224 and m['height'] == 224
)
```

### 3. Data Validation with Checksums

```cpp
// Validate entire dataset
TblV2Reader reader("dataset.tbl");
size_t corrupted_samples = 0;

for (size_t i = 0; i < reader.num_samples(); ++i) {
    try {
        auto sample = reader.read_sample(i);  // Validates CRC32
    } catch (const CRCMismatchError& e) {
        std::cerr << "Sample " << i << " corrupted: " << e.what() << std::endl;
        corrupted_samples++;
    }
}

std::cout << "Validation complete: " << corrupted_samples
          << " corrupted samples found" << std::endl;
```

### 4. Multi-Resolution Training

```python
# Load different resolutions for progressive training
loader_128 = turboloader.DataLoader(
    'dataset.tbl',
    filter_fn=lambda m: m['width'] == 128 and m['height'] == 128
)

loader_224 = turboloader.DataLoader(
    'dataset.tbl',
    filter_fn=lambda m: m['width'] == 224 and m['height'] == 224
)

loader_512 = turboloader.DataLoader(
    'dataset.tbl',
    filter_fn=lambda m: m['width'] == 512 and m['height'] == 512
)

# Progressive training: 128 → 224 → 512
train_epochs(model, loader_128, epochs=10)
train_epochs(model, loader_224, epochs=10)
train_epochs(model, loader_512, epochs=10)
```

## Advanced Features

### Parallel Conversion

Convert large TAR files using multiple threads:

```bash
# Use 16 worker threads for conversion
tar_to_tbl imagenet.tar imagenet.tbl --workers 16

# Expected speedup: ~12x (limited by I/O)
```

### Compression Level Tuning

```cpp
// Fast compression (default, LZ4 level 1)
TblV2Writer writer_fast("output.tbl", CompressionType::LZ4, 1);

// Balanced compression (LZ4 level 9, slower but smaller)
TblV2Writer writer_balanced("output.tbl", CompressionType::LZ4, 9);

// Maximum compression (future: Zstandard)
TblV2Writer writer_max("output.tbl", CompressionType::ZSTD, 19);
```

### Custom Metadata Schemas

```python
# Protobuf metadata for structured data
import turboloader
import sample_pb2  # Generated from .proto file

metadata = sample_pb2.DatasetMetadata()
metadata.name = "ImageNet"
metadata.version = 2012

for i, sample in enumerate(samples):
    s = metadata.samples.add()
    s.id = i
    s.class_label = sample['class']
    s.bbox.CopyFrom(sample['bbox'])

turboloader.convert_tar_to_tbl(
    'input.tar',
    'output.tbl',
    metadata=metadata.SerializeToString(),
    metadata_format='protobuf'
)
```

## Troubleshooting

### Issue: Conversion slower than expected

**Symptoms:**
- Throughput < 3,000 img/s
- High CPU usage during conversion

**Solutions:**

1. Use more workers:
```bash
tar_to_tbl input.tar output.tbl --workers 16
```

2. Use faster storage (NVMe SSD):
```bash
# Check I/O speed
dd if=/dev/zero of=test.dat bs=1M count=10000
```

3. Disable compression for fastest speed:
```bash
tar_to_tbl input.tar output.tbl --no-compression  # Uncompressed mode
```

### Issue: TBL v2 file larger than expected

**Symptoms:**
- File not 40-60% smaller than TAR
- Compression ratio < 1.5x

**Cause:**
- Images already heavily compressed (JPEG quality 95+)
- Small images (compression overhead)

**Check:**
```bash
# Analyze compression ratio
tar_to_tbl input.tar output.tbl --verbose

# Expected output:
# Compression ratio: 1.8:1 (good)
# Compression ratio: 1.1:1 (images already compressed)
```

### Issue: CRC32 validation errors

**Symptoms:**
- CRCMismatchError during read_sample()
- Corrupted samples

**Cause:**
- Disk corruption
- Incomplete file transfer
- Bad storage media

**Solution:**
```bash
# Re-convert from original TAR
tar_to_tbl original.tar dataset_new.tbl

# Validate checksum
md5sum dataset.tbl
```

### Issue: Out of memory during conversion

**Symptoms:**
- OOM killer during tar_to_tbl
- System hang during conversion

**Cause:**
- Insufficient system memory
- Very large individual samples

**Solution:**
```bash
# Ensure using latest version with streaming writer
pip install --upgrade turboloader

# Reduce number of parallel workers if needed
tar_to_tbl input.tar output.tbl --workers 4
```

## Creating TBL v2 Datasets

To create TBL v2 datasets from existing data:

```bash
# Convert from TAR source
tar_to_tbl original.tar dataset_v2.tbl

# With parallel processing for faster conversion
tar_to_tbl original.tar dataset_v2.tbl --workers 8
```

**Python API:**
```python
import turboloader

# Load TBL v2 files
loader = turboloader.DataLoader('dataset.tbl', batch_size=64)
```

## Best Practices

1. **Convert Once, Use Many Times**: TBL v2 conversion takes time (4.8k img/s), but saves 40-60% storage permanently

2. **Use for Large Datasets**: Benefits are most noticeable with >10 GB datasets

3. **Store on Fast Storage**: NVMe SSD recommended for conversion and reading

4. **Enable Checksums in Production**: Validates data integrity (slight overhead worth it)

5. **Cache Dimensions**: Use dimension filtering to avoid unnecessary decoding

6. **Parallel Conversion**: Use --workers flag for faster conversion on multi-core systems

## Performance Comparison

| Operation | TAR | TBL v2 (LZ4) |
|-----------|-----|--------------|
| **Sequential Read** | 8,672 img/s | 4,950 img/s |
| **Random Read** | 53 img/s | 4,800 img/s |
| **File Size** | 100 GB | 45-55 GB |
| **Conversion Speed** | N/A | 4,875 img/s |
| **Write Memory** | Variable | O(1) |
| **Data Integrity** | ❌ | ✅ (CRC32/16) |
| **Cached Dimensions** | ❌ | ✅ |
| **Compression** | ❌ | ✅ (LZ4) |

## Code Locations

- **Format Spec**: `src/formats/tbl_v2_format.hpp`
- **Reader**: `src/readers/tbl_v2_reader.hpp`
- **Writer**: `src/writers/tbl_v2_writer.hpp`
- **Converter**: `tools/tar_to_tbl.cpp`
- **Tests**: `tests/test_tbl_v2_format.cpp`
- **LZ4 Integration**: `src/compression/lz4_compressor.hpp`

## See Also

- [Architecture Documentation](../architecture.md) - TBL v2 pipeline design
- [Performance Benchmarks](../benchmarks/index.md) - Conversion throughput analysis
- [AVX-512 SIMD Guide](avx512-simd.md) - SIMD optimizations
- [CHANGELOG](../../CHANGELOG.md) - Version history and migration guide
