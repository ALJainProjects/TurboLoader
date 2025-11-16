# TurboLoader Integration Guide

How to integrate TurboLoader with PyTorch and TensorFlow training pipelines.

---

## PyTorch Integration

### Basic Integration

```python
import sys
sys.path.insert(0, 'build/python')
import turboloader

import torch
import torch.nn as nn
import torch.optim as optim

# Create TurboLoader pipeline
pipeline = turboloader.Pipeline(
    tar_paths=['/data/imagenet_train.tar'],
    num_workers=8,
    decode_jpeg=True
)

# Your model
model = torchvision.models.resnet50().cuda()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(100):
    pipeline.start()

    while True:
        batch = pipeline.next_batch(256)
        if len(batch) == 0:
            break

        # Convert to PyTorch tensors
        images = []
        labels = []
        for sample in batch:
            img = sample.get_image()  # NumPy (H, W, C)
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            images.append(img_tensor)
            # Extract label from filename or metadata
            labels.append(get_label(sample))

        images = torch.stack(images).cuda()
        labels = torch.tensor(labels).cuda()

        # Training step
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    pipeline.stop()
```

### With Data Augmentation

```python
import torchvision.transforms as T

transform = T.Compose([
    T.RandomResizedCrop(224),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# In training loop
for sample in batch:
    img_np = sample.get_image()
    img_pil = Image.fromarray(img_np.astype('uint8'))
    img_tensor = transform(img_pil)
```

### Custom Dataset Wrapper

```python
class TurboLoaderDataset:
    def __init__(self, tar_paths, num_workers=4, transform=None):
        self.pipeline = turboloader.Pipeline(
            tar_paths=tar_paths,
            num_workers=num_workers,
            decode_jpeg=True
        )
        self.transform = transform
        self.pipeline.start()

    def __iter__(self):
        while True:
            batch = self.pipeline.next_batch(1)
            if len(batch) == 0:
                self.pipeline.stop()
                self.pipeline.start()
                break

            sample = batch[0]
            img = sample.get_image()
            if self.transform:
                img = self.transform(Image.fromarray(img.astype('uint8')))
            yield img

    def __del__(self):
        self.pipeline.stop()
```

---

## TensorFlow Integration

```python
import tensorflow as tf
from tensorflow import keras

import sys
sys.path.insert(0, 'build/python')
import turboloader

# Create pipeline
pipeline = turboloader.Pipeline(
    tar_paths=['/data/imagenet_train.tar'],
    num_workers=8,
    decode_jpeg=True
)

# Model
model = keras.applications.ResNet50(weights=None, classes=1000)
model.compile(
    optimizer='sgd',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Training loop
for epoch in range(100):
    pipeline.start()

    while True:
        batch = pipeline.next_batch(256)
        if len(batch) == 0:
            break

        # Convert to TF tensors
        images = []
        labels = []
        for sample in batch:
            img = sample.get_image()
            images.append(img)
            labels.append(get_label(sample))

        images = tf.constant(images, dtype=tf.float32) / 255.0
        labels = tf.constant(labels, dtype=tf.int32)

        # Training step
        model.train_on_batch(images, labels)

    pipeline.stop()
```

---

## See Also

- [API Documentation](API.md)
- [Performance Tuning](PERFORMANCE.md)
