# Transform Library


TurboLoader includes 24 transforms (19 per-image SIMD transforms + 5 batch
augmentations). The authoritative list is `turboloader.list_transforms()`.

### Core Transforms
- **Resize** - Bilinear/Bicubic/Lanczos interpolation
- **Normalize** - Mean/std normalization with SIMD
- **CenterCrop** - Center region extraction
- **RandomCrop** - Random crop with padding

### Augmentation Transforms
- **RandomHorizontalFlip** - SIMD horizontal flip
- **RandomVerticalFlip** - SIMD vertical flip
- **ColorJitter** - Brightness/contrast/saturation/hue
- **RandomRotation** - Arbitrary angle rotation
- **GaussianBlur** - Separable convolution
- **RandomErasing** - Cutout augmentation
- **Pad** - Border padding (CONSTANT/EDGE/REFLECT)

### Advanced Transforms
- **RandomPosterize** - Bit-depth reduction
- **RandomSolarize** - Threshold inversion
- **RandomPerspective** - Perspective warp
- **AutoAugment** - Learned policies (ImageNet/CIFAR10/SVHN)

### Batch Augmentations
- **MixUp**, **CutMix**, **Mosaic**, **RandAugment**, **GridMask**

### Tensor Conversion
- **ToTensor** - PyTorch CHW or TensorFlow HWC format

Full per-transform API and examples: [docs/api/transforms.md](api/transforms.md).
