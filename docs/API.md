# TurboLoader API Documentation


> **Note**: Performance claims in this documentation are based on preliminary benchmarks on synthetic datasets. 
> Actual performance will vary based on hardware, dataset characteristics, and workload. 
> We recommend running benchmarks on your specific use case.



Complete API reference for both C++ and Python interfaces.

---

## Table of Contents

- [C++ API](#c-api)
  - [Pipeline Class](#pipeline-class)
  - [Configuration](#configuration)
  - [Sample Structure](#sample-structure)
- [Python API](#python-api)
  - [Pipeline Class](#python-pipeline-class)
  - [Sample Objects](#python-sample-objects)
- [Examples](#examples)

---

## C++ API

### Pipeline Class

**Header**: `#include <turboloader/pipeline/pipeline.hpp>`

**Namespace**: `turboloader`

#### Constructor

```cpp
Pipeline(const std::vector<std::string>& tar_paths, const Config& config);
```

**Parameters**:
- `tar_paths`: Vector of TAR file paths to load
- `config`: Pipeline configuration (see [Configuration](#configuration))

**Example**:
```cpp
#include <turboloader/pipeline/pipeline.hpp>

using namespace turboloader;

Pipeline::Config config{
    .num_workers = 8,
    .queue_size = 256,
    .prefetch_factor = 2,
    .shuffle = false,
    .decode_jpeg = true
};

Pipeline pipeline({"/data/train.tar"}, config);
```

---

#### Methods

##### `void start()`

Starts the data loading pipeline.

**Usage**:
```cpp
pipeline.start();
```

**Note**: Must be called before `next_batch()`.

---

##### `void stop()`

Stops the pipeline and releases resources.

**Usage**:
```cpp
pipeline.stop();
```

**Note**: Pipeline can be restarted with `start()` for next epoch.

---

##### `std::vector<Sample> next_batch(size_t batch_size)`

Fetches the next batch of samples.

**Parameters**:
- `batch_size`: Number of samples to fetch

**Returns**: Vector of `Sample` objects (empty if end of epoch)

**Usage**:
```cpp
auto batch = pipeline.next_batch(32);
if (batch.empty()) {
    // End of epoch
    pipeline.stop();
}
```

---

##### `void reset()`

Resets pipeline to beginning of dataset.

**Usage**:
```cpp
pipeline.reset();
```

---

##### `size_t total_samples() const`

Returns total number of samples across all TAR files.

**Returns**: Total sample count

**Usage**:
```cpp
size_t total = pipeline.total_samples();
std::cout << "Dataset has " << total << " samples\n";
```

---

### Configuration

```cpp
struct Pipeline::Config {
    size_t num_workers{4};          // Number of worker threads
    size_t queue_size{256};          // Internal queue size
    size_t prefetch_factor{2};       // Prefetch batches per worker
    bool shuffle{false};             // Shuffle samples
    size_t shuffle_buffer_size{1000}; // Shuffle buffer size
    bool decode_jpeg{false};         // Enable JPEG decoding

    // Transform options
    bool enable_resize{false};       // Enable image resize
    int resize_width{224};           // Target width
    int resize_height{224};          // Target height
    bool enable_normalize{false};    // Enable normalization
};
```

**Field descriptions**:

- **`num_workers`**: Number of parallel worker threads
  - Recommended: `num_cpu_cores` or slightly less
  - Higher = more parallelism, but diminishing returns

- **`queue_size`**: Size of internal sample queue
  - Larger = more buffering, but more memory
  - Recommended: 256-512

- **`prefetch_factor`**: Batches to prefetch per worker
  - Larger = better overlap, but more memory
  - Recommended: 2-4

- **`shuffle`**: Whether to shuffle samples
  - Uses shuffle buffer (not full dataset shuffle)

- **`decode_jpeg`**: Enable automatic JPEG decoding
  - Uses libjpeg-turbo (SIMD optimized)
  - Decoded images available via `Sample::width/height/channels`

- **`enable_resize`**: Enable image resizing
  - Requires `decode_jpeg = true`

---

### Sample Structure

```cpp
struct Sample {
    std::unordered_map<std::string, std::vector<uint8_t>> data;
    size_t index{0};

    // Decoded image data (if decode_jpeg enabled)
    int width{0};
    int height{0};
    int channels{0};
};
```

**Fields**:

- **`data`**: Map of file extensions to raw bytes
  - Example: `data[".jpg"]` contains JPEG bytes
  - Example: `data[".json"]` contains metadata

- **`index`**: Sample index in dataset

- **`width/height/channels`**: Decoded image dimensions (if `decode_jpeg` enabled)

**Accessing data**:
```cpp
auto batch = pipeline.next_batch(32);
for (const auto& sample : batch) {
    // Raw JPEG bytes
    const auto& jpeg_bytes = sample.data.at(".jpg");

    // Decoded image (if decode_jpeg enabled)
    std::cout << "Image size: " << sample.width << "x" << sample.height << "\n";
}
```

---

## Python API

### Python Pipeline Class

**Module**: `turboloader`

**Import**:
```python
import sys
sys.path.insert(0, 'build/python')
import turboloader
```

---

#### Constructor

```python
Pipeline(tar_paths, num_workers=4, decode_jpeg=True)
```

**Parameters**:
- `tar_paths` (list[str]): List of TAR file paths
- `num_workers` (int): Number of worker threads (default: 4)
- `decode_jpeg` (bool): Enable JPEG decoding (default: True)

**Example**:
```python
pipeline = turboloader.Pipeline(
    tar_paths=['/data/train.tar'],
    num_workers=8,
    decode_jpeg=True
)
```

---

#### Methods

##### `start()`

Starts the pipeline.

**Usage**:
```python
pipeline.start()
```

---

##### `stop()`

Stops the pipeline.

**Usage**:
```python
pipeline.stop()
```

**Note**: Always call `stop()` when done to release resources.

---

##### `next_batch(batch_size)`

Fetches next batch of samples.

**Parameters**:
- `batch_size` (int): Number of samples to fetch

**Returns**: List of `Sample` objects (empty list if end of epoch)

**Usage**:
```python
batch = pipeline.next_batch(32)
if len(batch) == 0:
    # End of epoch
    pipeline.stop()
```

---

### Python Sample Objects

Each sample in a batch has the following methods:

##### `get_image()`

Returns decoded image as NumPy array.

**Returns**: `numpy.ndarray` with shape `(height, width, channels)`

**Usage**:
```python
batch = pipeline.next_batch(32)
for sample in batch:
    img = sample.get_image()  # NumPy array (H, W, C)
    print(f"Image shape: {img.shape}")
```

**Note**: Requires `decode_jpeg=True` in Pipeline constructor.

---

##### `get_data(extension)`

Returns raw bytes for given file extension.

**Parameters**:
- `extension` (str): File extension (e.g., ".jpg", ".json")

**Returns**: `bytes` object

**Usage**:
```python
sample = batch[0]
jpeg_bytes = sample.get_data(".jpg")
json_bytes = sample.get_data(".json")
```

---

## Examples

### C++ Example - Basic Usage

```cpp
#include <iostream>
#include <turboloader/pipeline/pipeline.hpp>

using namespace turboloader;

int main() {
    // Configure pipeline
    Pipeline::Config config{
        .num_workers = 8,
        .queue_size = 256,
        .decode_jpeg = true
    };

    // Create pipeline
    Pipeline pipeline({"/data/train.tar"}, config);
    pipeline.start();

    // Training loop
    int epoch = 0;
    while (epoch < 10) {
        int samples_processed = 0;

        while (true) {
            auto batch = pipeline.next_batch(32);
            if (batch.empty()) break;

            // Process batch
            for (const auto& sample : batch) {
                std::cout << "Image: " << sample.width << "x"
                         << sample.height << "\n";
            }

            samples_processed += batch.size();
        }

        std::cout << "Epoch " << epoch << ": "
                  << samples_processed << " samples\n";

        pipeline.stop();
        pipeline.start();  // Restart for next epoch
        epoch++;
    }

    pipeline.stop();
    return 0;
}
```

---

### Python Example - PyTorch Integration

```python
import sys
sys.path.insert(0, 'build/python')
import turboloader

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Create pipeline
pipeline = turboloader.Pipeline(
    tar_paths=['/data/train.tar'],
    num_workers=8,
    decode_jpeg=True
)

# Model
model = nn.Sequential(
    nn.Conv2d(3, 64, 3),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.Linear(64, 10)
).cuda()

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(10):
    pipeline.start()

    while True:
        batch = pipeline.next_batch(32)
        if len(batch) == 0:
            break

        # Convert to PyTorch tensors
        images = []
        for sample in batch:
            img = sample.get_image()  # NumPy (H, W, C)
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            images.append(img_tensor)

        images = torch.stack(images).cuda()
        labels = torch.randint(0, 10, (len(images),)).cuda()

        # Forward + backward
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    pipeline.stop()
    print(f"Epoch {epoch} complete")
```

---

### Python Example - Iterator Pattern

```python
import sys
sys.path.insert(0, 'build/python')
import turboloader

def turboloader_iterator(tar_paths, batch_size, num_workers=4):
    """Create a Python iterator from TurboLoader pipeline"""
    pipeline = turboloader.Pipeline(
        tar_paths=tar_paths,
        num_workers=num_workers,
        decode_jpeg=True
    )
    pipeline.start()

    try:
        while True:
            batch = pipeline.next_batch(batch_size)
            if len(batch) == 0:
                break
            yield batch
    finally:
        pipeline.stop()

# Usage
for batch in turboloader_iterator(['/data/train.tar'], batch_size=32):
    for sample in batch:
        img = sample.get_image()
        print(f"Processing image: {img.shape}")
```

---

### Python Example - Custom Transforms

```python
import sys
sys.path.insert(0, 'build/python')
import turboloader

import torch
import torchvision.transforms as T
from PIL import Image

# Define transforms
transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

pipeline = turboloader.Pipeline(
    tar_paths=['/data/train.tar'],
    num_workers=8,
    decode_jpeg=True
)
pipeline.start()

batch = pipeline.next_batch(32)
transformed_images = []

for sample in batch:
    # Get NumPy image
    img_np = sample.get_image()

    # Convert to PIL
    img_pil = Image.fromarray(img_np.astype('uint8'))

    # Apply transforms
    img_tensor = transform(img_pil)
    transformed_images.append(img_tensor)

images = torch.stack(transformed_images)
print(f"Batch shape: {images.shape}")  # (32, 3, 224, 224)

pipeline.stop()
```

---

## Performance Tips

### Optimal Worker Count

```python
import multiprocessing

# Rule of thumb: num_workers = CPU cores
optimal_workers = multiprocessing.cpu_count()

pipeline = turboloader.Pipeline(
    tar_paths=['/data/train.tar'],
    num_workers=optimal_workers,
    decode_jpeg=True
)
```

### Batch Size Selection

Larger batches = higher throughput (up to a point):

| Batch Size | Throughput | Notes |
|------------|------------|-------|
| 8 | high throughput | Underutilized |
| 32 | high throughput | Good ✅ |
| 64 | high throughput | Optimal |
| 128 | high throughput | Marginal gains |

Recommendation: **32-64** for balanced throughput/latency.

### Memory Management

```python
# Always stop pipeline when done
pipeline.start()
try:
    # ... process data ...
    pass
finally:
    pipeline.stop()  # Releases memory
```

---

## Error Handling

### C++

```cpp
try {
    Pipeline pipeline({"/data/train.tar"}, config);
    pipeline.start();

    auto batch = pipeline.next_batch(32);
    // ... process batch ...

    pipeline.stop();
} catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
}
```

### Python

```python
try:
    pipeline = turboloader.Pipeline(
        tar_paths=['/data/train.tar'],
        num_workers=8,
        decode_jpeg=True
    )
    pipeline.start()

    batch = pipeline.next_batch(32)
    # ... process batch ...

except Exception as e:
    print(f"Error: {e}")
finally:
    pipeline.stop()
```

---

## See Also

- [Architecture Guide](ARCHITECTURE.md) - How TurboLoader works internally
- [Integration Guide](INTEGRATION.md) - PyTorch/TensorFlow integration patterns
- [Performance Tuning](PERFORMANCE.md) - Optimization tips
- [Benchmarks](../benchmarks/README.md) - Performance comparisons
