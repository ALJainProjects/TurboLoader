# TensorFlow Integration Guide

Complete guide for integrating TurboLoader with TensorFlow and Keras for training and inference.

## Table of Contents

- [Quick Start](#quick-start)
- [Basic Integration](#basic-integration)
- [Keras Model Training](#keras-model-training)
- [Custom Training Loop](#custom-training-loop)
- [Multi-GPU Training](#multi-gpu-training)
- [TF Data Pipeline Comparison](#tf-data-pipeline-comparison)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

Use TurboLoader with TensorFlow in 3 steps:

```python
import turboloader
import tensorflow as tf

# 1. Create transforms
transforms = turboloader.Compose([
    turboloader.Resize(224, 224),
    turboloader.ImageNetNormalize(),
    turboloader.ToTensor(turboloader.TensorFormat.TENSORFLOW_HWC)
])

# 2. Create TurboLoader
loader = turboloader.DataLoader(
    'imagenet_train.tar',
    batch_size=256,
    num_workers=8,
    shuffle=True
)

# 3. Convert to TensorFlow dataset
def generator():
    for batch in loader:
        images = []
        labels = []
        for sample in batch:
            img = transforms.apply(sample['image'])
            images.append(img)
            labels.append(sample['label'])
        yield tf.stack(images), tf.constant(labels)

dataset = tf.data.Dataset.from_generator(
    generator,
    output_signature=(
        tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.int32)
    )
)

# Train model
model.fit(dataset, epochs=10)
```

---

## Basic Integration

### Convert NumPy to TensorFlow Tensors

TurboLoader returns NumPy arrays that easily convert to TensorFlow tensors:

```python
import turboloader
import tensorflow as tf
import numpy as np

loader = turboloader.DataLoader('data.tar', batch_size=32, num_workers=4)

for batch in loader:
    for sample in batch:
        # NumPy array (H, W, C) uint8
        image_np = sample['image']

        # Convert to TensorFlow tensor (H, W, C) float32
        image_tensor = tf.convert_to_tensor(image_np, dtype=tf.float32)

        # Normalize to [0, 1]
        image_tensor = image_tensor / 255.0
```

### Using ToTensor Transform

Simplify with `ToTensor` transform:

```python
# HWC format (TensorFlow default)
to_tensor = turboloader.ToTensor(turboloader.TensorFormat.TENSORFLOW_HWC)

for batch in loader:
    for sample in batch:
        # Already in HWC format
        image_hwc = to_tensor.apply(sample['image'])  # (H, W, C)

        # Convert to TensorFlow tensor
        image_tensor = tf.convert_to_tensor(image_hwc, dtype=tf.float32) / 255.0
```

---

## Keras Model Training

### High-Level API with model.fit()

```python
import turboloader
import tensorflow as tf
from tensorflow import keras

# Create model
model = keras.applications.ResNet50(
    include_top=True,
    weights=None,
    input_shape=(224, 224, 3),
    classes=1000
)

model.compile(
    optimizer='sgd',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Transforms
transforms = turboloader.Compose([
    turboloader.Resize(256, 256),
    turboloader.RandomCrop(224, 224),
    turboloader.RandomHorizontalFlip(0.5),
    turboloader.ColorJitter(0.2, 0.2, 0.2, 0.1),
    turboloader.ImageNetNormalize(),
    turboloader.ToTensor(turboloader.TensorFormat.TENSORFLOW_HWC)
])

# TurboLoader
loader = turboloader.DataLoader(
    'imagenet_train.tar',
    batch_size=256,
    num_workers=8,
    shuffle=True
)

# Generator function
def data_generator():
    for batch in loader:
        images = []
        labels = []

        for sample in batch:
            img = transforms.apply(sample['image'])
            images.append(img)
            labels.append(sample['label'])

        # Stack into batch
        images_batch = tf.stack([tf.convert_to_tensor(img, dtype=tf.float32) for img in images])
        labels_batch = tf.constant(labels, dtype=tf.int32)

        yield images_batch, labels_batch

# Create TensorFlow dataset
dataset = tf.data.Dataset.from_generator(
    data_generator,
    output_signature=(
        tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.int32)
    )
)

# Train
history = model.fit(dataset, epochs=90, steps_per_epoch=5000)
```

### With Validation Data

```python
# Validation transforms (deterministic)
val_transforms = turboloader.Compose([
    turboloader.Resize(256, 256),
    turboloader.CenterCrop(224, 224),
    turboloader.ImageNetNormalize(),
    turboloader.ToTensor(turboloader.TensorFormat.TENSORFLOW_HWC)
])

# Validation loader
val_loader = turboloader.DataLoader(
    'imagenet_val.tar',
    batch_size=256,
    num_workers=8,
    shuffle=False
)

def val_generator():
    for batch in val_loader:
        images = []
        labels = []
        for sample in batch:
            img = val_transforms.apply(sample['image'])
            images.append(img)
            labels.append(sample['label'])

        images_batch = tf.stack([tf.convert_to_tensor(img, dtype=tf.float32) for img in images])
        labels_batch = tf.constant(labels, dtype=tf.int32)
        yield images_batch, labels_batch

val_dataset = tf.data.Dataset.from_generator(
    val_generator,
    output_signature=(
        tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.int32)
    )
)

# Train with validation
history = model.fit(
    dataset,
    epochs=90,
    steps_per_epoch=5000,
    validation_data=val_dataset,
    validation_steps=200
)
```

---

## Custom Training Loop

For more control, use TensorFlow's GradientTape:

```python
import turboloader
import tensorflow as tf

# Model
model = tf.keras.applications.ResNet50(weights=None, classes=1000)

# Loss and optimizer
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)

# Metrics
train_loss = tf.keras.metrics.Mean()
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

# Transforms
transforms = turboloader.Compose([
    turboloader.Resize(256, 256),
    turboloader.RandomCrop(224, 224),
    turboloader.RandomHorizontalFlip(0.5),
    turboloader.ImageNetNormalize(),
    turboloader.ToTensor(turboloader.TensorFormat.TENSORFLOW_HWC)
])

# DataLoader
loader = turboloader.DataLoader(
    'imagenet_train.tar',
    batch_size=256,
    num_workers=8,
    shuffle=True
)

# Training function
@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_fn(labels, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)

    return loss

# Training loop
for epoch in range(90):
    print(f'Epoch {epoch + 1}/90')

    train_loss.reset_states()
    train_accuracy.reset_states()

    for batch_idx, batch in enumerate(loader):
        # Process batch
        images = []
        labels = []

        for sample in batch:
            img = transforms.apply(sample['image'])
            images.append(tf.convert_to_tensor(img, dtype=tf.float32))
            labels.append(sample['label'])

        images = tf.stack(images)
        labels = tf.constant(labels, dtype=tf.int32)

        # Training step
        loss = train_step(images, labels)

        if batch_idx % 100 == 0:
            print(f'  Batch {batch_idx}: Loss {train_loss.result():.4f}, '
                  f'Accuracy {train_accuracy.result():.4f}')

    print(f'Epoch {epoch + 1}: Loss {train_loss.result():.4f}, '
          f'Accuracy {train_accuracy.result():.4f}')
```

---

## Multi-GPU Training

### tf.distribute.MirroredStrategy

Train on multiple GPUs with minimal code changes:

```python
import turboloader
import tensorflow as tf

# Create strategy
strategy = tf.distribute.MirroredStrategy()

print(f'Number of devices: {strategy.num_replicas_in_sync}')

# Model creation inside strategy scope
with strategy.scope():
    model = tf.keras.applications.ResNet50(weights=None, classes=1000)

    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

# Transforms
transforms = turboloader.Compose([
    turboloader.Resize(256, 256),
    turboloader.RandomCrop(224, 224),
    turboloader.RandomHorizontalFlip(0.5),
    turboloader.ImageNetNormalize(),
    turboloader.ToTensor(turboloader.TensorFormat.TENSORFLOW_HWC)
])

# TurboLoader with distributed sharding
loader = turboloader.DataLoader(
    'imagenet_train.tar',
    batch_size=256,  # Per-GPU batch size
    num_workers=8,
    shuffle=True,
    enable_distributed=True  # Enable automatic sharding
)

def data_generator():
    for batch in loader:
        images = []
        labels = []
        for sample in batch:
            img = transforms.apply(sample['image'])
            images.append(img)
            labels.append(sample['label'])

        images_batch = tf.stack([tf.convert_to_tensor(img, dtype=tf.float32) for img in images])
        labels_batch = tf.constant(labels, dtype=tf.int32)
        yield images_batch, labels_batch

dataset = tf.data.Dataset.from_generator(
    data_generator,
    output_signature=(
        tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.int32)
    )
)

# Distribute dataset
dist_dataset = strategy.experimental_distribute_dataset(dataset)

# Train
model.fit(dist_dataset, epochs=90, steps_per_epoch=5000)
```

---

## TF Data Pipeline Comparison

### TensorFlow tf.data (Before)

```python
import tensorflow as tf

# Load from directory
dataset = tf.keras.utils.image_dataset_from_directory(
    'imagenet/train',
    labels='inferred',
    label_mode='int',
    batch_size=256,
    image_size=(224, 224)
)

# Apply augmentation
def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.resize(image, [256, 256])
    image = tf.image.random_crop(image, [224, 224, 3])
    return image, label

dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.prefetch(tf.data.AUTOTUNE)

# Train
model.fit(dataset, epochs=90)
```

### TurboLoader (After)

```python
import turboloader
import tensorflow as tf

# Load from TAR (faster)
loader = turboloader.DataLoader(
    'imagenet_train.tar',
    batch_size=256,
    num_workers=8,
    shuffle=True
)

# SIMD-accelerated augmentation (faster)
transforms = turboloader.Compose([
    turboloader.Resize(256, 256),
    turboloader.RandomCrop(224, 224),
    turboloader.RandomHorizontalFlip(0.5),
    turboloader.ColorJitter(0.2, 0.2, 0.2, 0.1),
    turboloader.ImageNetNormalize()
])

def generator():
    for batch in loader:
        images = []
        labels = []
        for sample in batch:
            img = transforms.apply(sample['image'])
            images.append(tf.convert_to_tensor(img, dtype=tf.float32))
            labels.append(sample['label'])
        yield tf.stack(images), tf.constant(labels)

dataset = tf.data.Dataset.from_generator(generator, ...)

# Train
model.fit(dataset, epochs=90)
```

**Benefits:**
- **10-12x faster data loading** (TurboLoader vs tf.data)
- **SIMD-accelerated transforms** (2-5x faster than TF ops)
- **Lower CPU usage** (C++ implementation vs Python)

---

## Best Practices

### 1. Use Appropriate Tensor Format

```python
# TensorFlow uses HWC format
to_tensor = turboloader.ToTensor(turboloader.TensorFormat.TENSORFLOW_HWC)
```

### 2. Batch Processing

Process batches together for efficiency:

```python
def batch_generator():
    for batch in loader:
        # Process entire batch at once
        images = [transforms.apply(s['image']) for s in batch]
        labels = [s['label'] for s in batch]

        # Stack efficiently
        images_batch = tf.stack([tf.convert_to_tensor(img, dtype=tf.float32) for img in images])
        labels_batch = tf.constant(labels, dtype=tf.int32)

        yield images_batch, labels_batch
```

### 3. Prefetch with TensorFlow

Combine TurboLoader with TF prefetch:

```python
dataset = tf.data.Dataset.from_generator(generator, ...)
dataset = dataset.prefetch(tf.data.AUTOTUNE)  # Prefetch to GPU
```

### 4. Mixed Precision Training

Enable for faster training:

```python
from tensorflow.keras import mixed_precision

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# Model will use float16 for compute, float32 for variables
model = tf.keras.applications.ResNet50(...)
```

---

## Troubleshooting

### Issue: "Invalid argument" error with from_generator

**Problem**: Dataset signature doesn't match actual data.

**Solution**: Ensure output_signature matches exactly:

```python
dataset = tf.data.Dataset.from_generator(
    generator,
    output_signature=(
        tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),  # (batch, H, W, C)
        tf.TensorSpec(shape=(None,), dtype=tf.int32)  # (batch,)
    )
)
```

### Issue: Slow data loading

**Problem**: Generator is slower than expected.

**Solutions**:
1. Increase `num_workers` in TurboLoader
2. Use larger `batch_size` if GPU memory allows
3. Convert TAR to TBL format for faster loading
4. Use `dataset.prefetch(tf.data.AUTOTUNE)`

### Issue: Memory leak

**Problem**: Memory usage grows over time.

**Solution**: Explicitly delete batches:

```python
def generator():
    for batch in loader:
        images, labels = process_batch(batch)
        yield images, labels
        del batch, images, labels  # Explicit cleanup
```

### Issue: Different results on each epoch

**Problem**: Shuffle behavior inconsistent.

**Solution**: Control shuffle in TurboLoader:

```python
# Training: shuffle enabled
train_loader = turboloader.DataLoader(..., shuffle=True)

# Validation: shuffle disabled
val_loader = turboloader.DataLoader(..., shuffle=False)
```

---

## Additional Resources

- [Transforms API Reference](../api/transforms.md) - Complete transform documentation
- [TensorFlow Integration Tests](../../tests/test_tensorflow_integration.py) - Working examples
- [Benchmarks](../benchmarks/index.md) - Performance comparison vs tf.data
- [TensorFlow Official Docs](https://www.tensorflow.org/guide) - TensorFlow guides

---

## Example: Complete Training Script

```python
#!/usr/bin/env python3
"""TensorFlow ImageNet training with TurboLoader."""

import turboloader
import tensorflow as tf
from tensorflow import keras

# Model
model = keras.applications.ResNet50(weights=None, classes=1000)
model.compile(
    optimizer=keras.optimizers.SGD(learning_rate=0.1, momentum=0.9),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Transforms
transforms = turboloader.Compose([
    turboloader.Resize(256, 256),
    turboloader.RandomCrop(224, 224),
    turboloader.RandomHorizontalFlip(0.5),
    turboloader.ImageNetNormalize(),
    turboloader.ToTensor(turboloader.TensorFormat.TENSORFLOW_HWC)
])

# TurboLoader
loader = turboloader.DataLoader(
    'imagenet_train.tar',
    batch_size=256,
    num_workers=8,
    shuffle=True
)

# Generator
def generator():
    for batch in loader:
        images = [transforms.apply(s['image']) for s in batch]
        labels = [s['label'] for s in batch]

        images_batch = tf.stack([tf.convert_to_tensor(img, dtype=tf.float32) for img in images])
        labels_batch = tf.constant(labels, dtype=tf.int32)

        yield images_batch, labels_batch

# Dataset
dataset = tf.data.Dataset.from_generator(
    generator,
    output_signature=(
        tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.int32)
    )
)
dataset = dataset.prefetch(tf.data.AUTOTUNE)

# Train
history = model.fit(dataset, epochs=90, steps_per_epoch=5000)

print(f"Final accuracy: {history.history['accuracy'][-1]:.4f}")
```
