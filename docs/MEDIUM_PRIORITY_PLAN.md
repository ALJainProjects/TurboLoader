# TurboLoader v1.9.0 - Medium Priority Enhancements Plan

## Overview
This document outlines the implementation plan for medium-priority enhancements to TurboLoader.

---

## 1. Format Support Gaps

### 1.1 HDF5 Support
**Priority:** High (common in scientific computing)

**Implementation:**
- Add HDF5 reader using libhdf5
- Support datasets, groups, and attributes
- Memory-mapped access for large files
- Chunked reading for streaming

**Files to create/modify:**
- `src/readers/hdf5_reader.hpp` - HDF5 reader class
- `src/readers/hdf5_reader.cpp` - Implementation
- `src/python/turboloader_bindings.cpp` - Python bindings
- `CMakeLists.txt` - Add HDF5 dependency

**API Design:**
```python
# Python API
reader = turboloader.HDF5Reader("/path/to/data.h5")
reader.list_datasets()  # Returns dataset names
reader.read_dataset("images")  # Returns numpy array
reader.read_dataset("images", slice=(0, 100))  # Sliced read

# Integration with DataLoader
loader = turboloader.DataLoader(
    source="/path/to/data.h5",
    format="hdf5",
    dataset="train/images"
)
```

**Dependencies:**
- libhdf5 (brew install hdf5 / apt install libhdf5-dev)

---

### 1.2 COCO/Pascal VOC Annotation Support
**Priority:** High (standard object detection formats)

**Implementation:**
- COCO JSON parser for annotations
- Pascal VOC XML parser
- Bounding box, segmentation, keypoint support
- Category mapping

**Files to create/modify:**
- `src/formats/coco_parser.hpp` - COCO annotation parser
- `src/formats/voc_parser.hpp` - Pascal VOC parser
- `src/formats/annotation_types.hpp` - Common annotation types
- `src/python/turboloader_bindings.cpp` - Python bindings

**API Design:**
```python
# COCO format
dataset = turboloader.COCODataset(
    image_dir="/path/to/images",
    annotation_file="/path/to/annotations.json"
)

for image, annotations in dataset:
    # annotations contains bboxes, segmentation masks, etc.
    pass

# Pascal VOC format
dataset = turboloader.VOCDataset(
    root_dir="/path/to/VOCdevkit/VOC2012",
    split="train"  # train, val, trainval
)
```

**Data structures:**
```cpp
struct BoundingBox {
    float x, y, width, height;
    int category_id;
    float confidence;
};

struct Annotation {
    std::vector<BoundingBox> bboxes;
    std::vector<std::vector<float>> segmentation;  // Polygon points
    std::vector<float> keypoints;
};
```

---

### 1.3 TFRecord Support
**Priority:** Medium (TensorFlow ecosystem compatibility)

**Implementation:**
- TFRecord file reader (protobuf-based)
- Example/SequenceExample parsing
- Feature extraction (bytes, float, int64)
- Sharded file support

**Files to create/modify:**
- `src/readers/tfrecord_reader.hpp`
- `src/readers/tfrecord_reader.cpp`
- `src/formats/tf_example.proto` - TensorFlow Example protobuf
- `src/python/turboloader_bindings.cpp`

**API Design:**
```python
# Read TFRecords
reader = turboloader.TFRecordReader("/path/to/data.tfrecord")

# With DataLoader
loader = turboloader.DataLoader(
    source="/path/to/data-*.tfrecord",  # Sharded
    format="tfrecord",
    features={
        "image": turboloader.FixedLenFeature([], dtype="string"),
        "label": turboloader.FixedLenFeature([], dtype="int64"),
    }
)
```

**Dependencies:**
- protobuf (already likely installed)

---

### 1.4 Zarr Support
**Priority:** Medium (cloud-native array storage)

**Implementation:**
- Zarr v2 format reader
- Chunked array access
- Cloud storage backend support (S3, GCS)
- Compression codec support (blosc, zstd, lz4)

**Files to create/modify:**
- `src/readers/zarr_reader.hpp`
- `src/readers/zarr_reader.cpp`
- `src/python/turboloader_bindings.cpp`

**API Design:**
```python
# Local Zarr
reader = turboloader.ZarrReader("/path/to/data.zarr")
array = reader.read_array("images", chunks=[(0, 100)])

# Cloud Zarr
reader = turboloader.ZarrReader("s3://bucket/data.zarr")
```

**Dependencies:**
- blosc (compression)
- Existing S3/GCS readers for cloud support

---

## 2. io_uring on Linux

**Priority:** High (significant performance boost on Linux)

**Implementation:**
- Conditional compilation for Linux with io_uring
- Async I/O for file reads
- Batch submission of I/O requests
- Fallback to standard I/O on unsupported systems

**Files to create/modify:**
- `src/io/io_uring_reader.hpp` - io_uring wrapper
- `src/io/io_uring_reader.cpp` - Implementation
- `src/io/file_reader.hpp` - Abstract file reader interface
- `CMakeLists.txt` - Detect and link liburing

**API Design:**
```cpp
// C++ internal API
class IoUringReader : public FileReader {
public:
    IoUringReader(int queue_depth = 32);

    // Submit async read
    void submit_read(int fd, void* buf, size_t len, off_t offset);

    // Wait for completions
    std::vector<IoResult> wait_completions(int min_completions = 1);

    // Batch read multiple files
    void batch_read(const std::vector<ReadRequest>& requests);
};
```

```python
# Python - automatic backend selection
loader = turboloader.DataLoader(
    source="/path/to/data",
    io_backend="auto"  # auto, io_uring, mmap, standard
)

# Check if io_uring is available
print(turboloader.features())  # Shows io_uring: true/false
```

**Requirements:**
- Linux kernel 5.1+ for basic io_uring
- Linux kernel 5.6+ for full feature set
- liburing library

**Fallback strategy:**
1. Check kernel version at runtime
2. Fall back to mmap or standard I/O if unavailable
3. Compile-time detection for systems without liburing headers

---

## 3. Transform Composition API (Pipe Operator)

**Priority:** High (developer experience improvement)

**Implementation:**
- Overload `|` operator for Transform classes
- Chain transforms into ComposedTransforms
- Type-safe composition with compile-time checks

**Files to modify:**
- `src/transforms/transform.hpp` - Add operator overloads
- `src/python/turboloader_bindings.cpp` - Python `__or__` method

**API Design:**
```cpp
// C++ API
auto pipeline = Resize(224, 224) | RandomHorizontalFlip(0.5) | Normalize({0.485, 0.456, 0.406}, {0.229, 0.224, 0.225}) | ToTensor();

// Apply to image
auto result = pipeline.apply(image);
```

```python
# Python API
pipeline = (
    turboloader.Resize(224, 224) |
    turboloader.RandomHorizontalFlip(0.5) |
    turboloader.ImageNetNormalize() |
    turboloader.ToTensor()
)

# Equivalent to Compose([...])
result = pipeline.apply(image)

# Can also build incrementally
base = turboloader.Resize(224, 224) | turboloader.CenterCrop(224)
augmented = base | turboloader.RandomHorizontalFlip(0.5)
```

**Implementation details:**
```cpp
// In transform.hpp
class Transform {
public:
    // Pipe operator returns ComposedTransforms
    ComposedTransforms operator|(const Transform& other) const {
        return ComposedTransforms({*this, other});
    }
};

class ComposedTransforms : public Transform {
public:
    ComposedTransforms operator|(const Transform& other) const {
        auto new_transforms = transforms_;
        new_transforms.push_back(other);
        return ComposedTransforms(new_transforms);
    }
};
```

---

## 4. GPU Transform Pipeline

**Priority:** Medium (CUDA-accelerated transforms)

**Implementation:**
- CUDA kernels for common transforms
- GPU memory management (pinned memory, streams)
- Async transfer between CPU and GPU
- NPP (NVIDIA Performance Primitives) integration

**Files to create/modify:**
- `src/transforms/gpu/` - New directory for GPU transforms
- `src/transforms/gpu/cuda_transforms.cu` - CUDA kernels
- `src/transforms/gpu/gpu_pipeline.hpp` - GPU pipeline class
- `CMakeLists.txt` - CUDA compilation support

**Transforms to GPU-accelerate:**
1. Resize (bilinear, bicubic, Lanczos)
2. Normalize
3. ColorJitter
4. RandomCrop
5. RandomHorizontalFlip
6. GaussianBlur

**API Design:**
```python
# Create GPU pipeline
gpu_pipeline = turboloader.GPUPipeline(
    device=0,  # CUDA device
    transforms=[
        turboloader.Resize(224, 224),
        turboloader.RandomHorizontalFlip(0.5),
        turboloader.ImageNetNormalize(),
    ]
)

# Process batch on GPU
batch_gpu = gpu_pipeline.process_batch(images)  # Returns GPU tensor

# Or integrate with DataLoader
loader = turboloader.DataLoader(
    source="/path/to/data.tar",
    transforms=gpu_pipeline,
    prefetch_to_gpu=True
)
```

**Requirements:**
- CUDA Toolkit 11.0+
- cuDNN (optional, for convolution-based transforms)
- NPP library (included with CUDA)

**Fallback:**
- Graceful fallback to CPU transforms when CUDA unavailable
- Runtime detection of CUDA capability

---

## 5. Multi-platform Wheels

**Priority:** High (ease of installation)

**Implementation:**
- GitHub Actions CI/CD for wheel building
- cibuildwheel for cross-platform builds
- Pre-built wheels for:
  - Linux x86_64 (manylinux2014)
  - Linux aarch64 (manylinux2014)
  - macOS x86_64
  - macOS arm64 (Apple Silicon)
  - Windows x86_64

**Files to create/modify:**
- `.github/workflows/build-wheels.yml` - CI workflow
- `pyproject.toml` - cibuildwheel configuration
- `setup.py` - Platform-specific build options

**CI Configuration:**
```yaml
# .github/workflows/build-wheels.yml
name: Build Wheels

on:
  release:
    types: [published]
  workflow_dispatch:

jobs:
  build_wheels:
    name: Build wheels on ${{ matrix.os }}
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]

    steps:
      - uses: actions/checkout@v4

      - name: Install dependencies (Linux)
        if: runner.os == 'Linux'
        run: |
          sudo apt-get update
          sudo apt-get install -y libjpeg-turbo8-dev libpng-dev libwebp-dev libcurl4-openssl-dev liblz4-dev

      - name: Install dependencies (macOS)
        if: runner.os == 'macOS'
        run: |
          brew install jpeg-turbo libpng webp curl lz4

      - name: Build wheels
        uses: pypa/cibuildwheel@v2.16
        env:
          CIBW_BUILD: "cp38-* cp39-* cp310-* cp311-* cp312-*"
          CIBW_SKIP: "*-musllinux_*"
          CIBW_ARCHS_LINUX: "x86_64 aarch64"
          CIBW_ARCHS_MACOS: "x86_64 arm64"

      - uses: actions/upload-artifact@v3
        with:
          name: wheels
          path: ./wheelhouse/*.whl

  publish:
    needs: build_wheels
    runs-on: ubuntu-latest
    steps:
      - uses: actions/download-artifact@v3
      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          packages-dir: wheels/
```

**pyproject.toml additions:**
```toml
[tool.cibuildwheel]
build-verbosity = 1
test-command = "python -c \"import turboloader; print(turboloader.version())\""

[tool.cibuildwheel.linux]
before-all = "yum install -y libjpeg-turbo-devel libpng-devel libwebp-devel libcurl-devel lz4-devel"
manylinux-x86_64-image = "manylinux2014"
manylinux-aarch64-image = "manylinux2014"

[tool.cibuildwheel.macos]
before-all = "brew install jpeg-turbo libpng webp curl lz4"

[tool.cibuildwheel.windows]
before-build = "pip install delvewheel"
repair-wheel-command = "delvewheel repair -w {dest_dir} {wheel}"
```

---

## 6. Azure Blob Storage Support

**Priority:** Medium (enterprise cloud support)

**Implementation:**
- Azure Blob Storage client using Azure SDK for C++
- Support for block blobs and append blobs
- SAS token and connection string authentication
- Streaming reads for large blobs

**Files to create/modify:**
- `src/readers/azure_blob_reader.hpp`
- `src/readers/azure_blob_reader.cpp`
- `src/python/turboloader_bindings.cpp`
- `CMakeLists.txt` - Azure SDK dependency

**API Design:**
```python
# Using connection string
loader = turboloader.DataLoader(
    source="azure://container/path/to/data.tar",
    azure_connection_string=os.environ["AZURE_STORAGE_CONNECTION_STRING"]
)

# Using SAS token
loader = turboloader.DataLoader(
    source="https://account.blob.core.windows.net/container/data.tar",
    azure_sas_token="?sv=2021-06-08&ss=b&srt=co&sp=r..."
)

# Using managed identity (for Azure VMs)
loader = turboloader.DataLoader(
    source="azure://container/path/to/data.tar",
    azure_use_managed_identity=True
)
```

**Authentication methods:**
1. Connection string
2. SAS token
3. Account key
4. Managed identity (Azure AD)
5. Service principal

**Dependencies:**
- Azure SDK for C++ (azure-storage-blobs-cpp)

---

## Implementation Order

Based on dependencies and impact:

### Phase 1: Foundation (v1.9.0)
1. **Transform Composition API** - Quick win, improves DX
2. **Multi-platform Wheels** - Critical for adoption
3. **COCO/Pascal VOC Support** - High demand

### Phase 2: Performance (v1.10.0)
4. **io_uring on Linux** - Major performance boost
5. **HDF5 Support** - Scientific computing users

### Phase 3: Cloud & GPU (v1.11.0)
6. **Azure Blob Storage** - Enterprise cloud
7. **TFRecord Support** - TensorFlow ecosystem
8. **Zarr Support** - Cloud-native arrays

### Phase 4: GPU Acceleration (v2.0.0)
9. **GPU Transform Pipeline** - CUDA acceleration

---

## Estimated Complexity

| Feature | Files | LOC | Complexity |
|---------|-------|-----|------------|
| Transform Pipe Operator | 2 | ~100 | Low |
| Multi-platform Wheels | 3 | ~200 | Medium |
| COCO/VOC Support | 4 | ~800 | Medium |
| io_uring | 3 | ~500 | High |
| HDF5 Support | 3 | ~600 | Medium |
| Azure Blob Storage | 3 | ~700 | Medium |
| TFRecord Support | 4 | ~800 | Medium |
| Zarr Support | 3 | ~600 | Medium |
| GPU Transform Pipeline | 6 | ~2000 | High |

---

## Testing Strategy

Each feature requires:
1. Unit tests for core functionality
2. Integration tests with DataLoader
3. Performance benchmarks
4. Documentation and examples

Test files to create:
- `tests/test_hdf5.py`
- `tests/test_coco_voc.py`
- `tests/test_tfrecord.py`
- `tests/test_zarr.py`
- `tests/test_io_uring.py` (Linux only)
- `tests/test_pipe_operator.py`
- `tests/test_gpu_transforms.py` (CUDA only)
- `tests/test_azure_blob.py`
