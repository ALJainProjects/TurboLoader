"""
TurboLoader PyTorch Compatibility Layer

Provides drop-in replacement compatibility with PyTorch DataLoader.

Features:
1. PyTorchCompatibleLoader - Returns (images, labels) tuples like PyTorch
2. LabelExtractor - Automatic label extraction from filenames/paths
3. ImageFolderConverter - Convert ImageFolder datasets to TAR format
4. TransformAdapter - Bridge between torchvision and turboloader transforms

Usage:
    from turboloader.pytorch_compat import (
        PyTorchCompatibleLoader,
        ImageFolderConverter,
        FolderLabelExtractor
    )

    # Convert existing ImageFolder to TAR (one-time)
    converter = ImageFolderConverter()
    converter.convert('/path/to/imagenet/train', 'train.tar')

    # Use exactly like PyTorch DataLoader
    loader = PyTorchCompatibleLoader(
        'train.tar',
        batch_size=128,
        shuffle=True,
        num_workers=8,
        label_extractor=FolderLabelExtractor()
    )

    for images, labels in loader:
        # images: torch.Tensor (N, C, H, W)
        # labels: torch.Tensor (N,)
        outputs = model(images)
        loss = criterion(outputs, labels)
"""

import os
import sys
import json
import tarfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union, Any, Iterator
from abc import ABC, abstractmethod

import numpy as np

try:
    import torch
    from torch.utils.data import IterableDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

# Import turboloader components
try:
    from _turboloader import DataLoader as _DataLoaderBase
    import turboloader

    TURBOLOADER_AVAILABLE = True
except ImportError:
    TURBOLOADER_AVAILABLE = False


# =============================================================================
# LABEL EXTRACTORS
# =============================================================================


class LabelExtractor(ABC):
    """Base class for extracting labels from sample metadata."""

    @abstractmethod
    def extract(self, filename: str, metadata: Dict[str, Any]) -> int:
        """Extract label from filename and/or metadata.

        Args:
            filename: The filename/path of the sample (e.g., 'dog/image_001.jpg')
            metadata: Additional metadata dict from TurboLoader

        Returns:
            Integer label
        """
        pass

    def get_num_classes(self) -> Optional[int]:
        """Return number of classes if known, None otherwise."""
        return None

    def get_class_names(self) -> Optional[List[str]]:
        """Return list of class names if known, None otherwise."""
        return None


class FolderLabelExtractor(LabelExtractor):
    """Extract labels from folder structure (ImageFolder style).

    Expects paths like: class_name/image.jpg or class_name/subdir/image.jpg
    Labels are assigned based on sorted folder names.

    Example:
        cat/img001.jpg -> 0
        dog/img001.jpg -> 1
        fish/img001.jpg -> 2
    """

    def __init__(self, class_to_idx: Optional[Dict[str, int]] = None):
        """
        Args:
            class_to_idx: Optional mapping from class names to indices.
                         If None, will be built dynamically from first pass.
        """
        self._class_to_idx = class_to_idx or {}
        self._idx_to_class: Dict[int, str] = {}
        self._next_idx = 0

        if class_to_idx:
            self._idx_to_class = {v: k for k, v in class_to_idx.items()}
            self._next_idx = max(class_to_idx.values()) + 1

    def extract(self, filename: str, metadata: Dict[str, Any]) -> int:
        """Extract label from folder name."""
        # Handle various path formats
        parts = filename.replace("\\", "/").split("/")

        if len(parts) >= 2:
            class_name = parts[0]  # First directory is the class
        else:
            class_name = "unknown"

        # Get or assign index
        if class_name not in self._class_to_idx:
            self._class_to_idx[class_name] = self._next_idx
            self._idx_to_class[self._next_idx] = class_name
            self._next_idx += 1

        return self._class_to_idx[class_name]

    def get_num_classes(self) -> int:
        return len(self._class_to_idx)

    def get_class_names(self) -> List[str]:
        return [self._idx_to_class[i] for i in range(len(self._idx_to_class))]

    @property
    def class_to_idx(self) -> Dict[str, int]:
        return self._class_to_idx.copy()


class FilenamePatternExtractor(LabelExtractor):
    r"""Extract labels from filename patterns using regex.

    Example:
        Pattern: r'class_(\d+)_.*\.jpg'
        Filename: 'class_5_image_001.jpg' -> label 5
    """

    def __init__(self, pattern: str, group: int = 1):
        """
        Args:
            pattern: Regex pattern with capture group for label
            group: Which capture group contains the label (default: 1)
        """
        import re

        self._pattern = re.compile(pattern)
        self._group = group

    def extract(self, filename: str, metadata: Dict[str, Any]) -> int:
        match = self._pattern.search(filename)
        if match:
            return int(match.group(self._group))
        return 0


class MetadataLabelExtractor(LabelExtractor):
    """Extract labels from TurboLoader metadata dict.

    Useful when labels are stored in sidecar files or TAR metadata.
    """

    def __init__(self, key: str = "label", default: int = 0):
        """
        Args:
            key: Metadata key containing the label
            default: Default label if key not found
        """
        self._key = key
        self._default = default

    def extract(self, filename: str, metadata: Dict[str, Any]) -> int:
        return metadata.get(self._key, self._default)


class JSONSidecarExtractor(LabelExtractor):
    """Extract labels from JSON sidecar files.

    For each image, looks for a corresponding .json file with label info.
    """

    def __init__(self, label_key: str = "label", cache: bool = True):
        self._label_key = label_key
        self._cache = cache
        self._label_cache: Dict[str, int] = {}

    def extract(self, filename: str, metadata: Dict[str, Any]) -> int:
        # Check cache first
        if self._cache and filename in self._label_cache:
            return self._label_cache[filename]

        # Try to find JSON in metadata
        json_key = filename.rsplit(".", 1)[0] + ".json"
        if "json_data" in metadata:
            data = metadata["json_data"]
            label = data.get(self._label_key, 0)
        else:
            label = 0

        if self._cache:
            self._label_cache[filename] = label

        return label


class CallableLabelExtractor(LabelExtractor):
    """Use a custom callable for label extraction.

    Example:
        extractor = CallableLabelExtractor(
            lambda fn, meta: int(fn.split('_')[1])
        )
    """

    def __init__(self, func: Callable[[str, Dict], int]):
        self._func = func

    def extract(self, filename: str, metadata: Dict[str, Any]) -> int:
        return self._func(filename, metadata)


# =============================================================================
# PYTORCH COMPATIBLE LOADER
# =============================================================================


class PyTorchCompatibleLoader:
    """TurboLoader with PyTorch DataLoader-compatible interface.

    Returns (images, labels) tuples exactly like PyTorch DataLoader,
    making it a true drop-in replacement.

    Example:
        # Before (PyTorch)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        for images, labels in loader:
            ...

        # After (TurboLoader - same interface!)
        loader = PyTorchCompatibleLoader('data.tar', batch_size=32, shuffle=True)
        for images, labels in loader:
            ...
    """

    def __init__(
        self,
        data_path: str,
        batch_size: int = 32,
        shuffle: bool = False,
        num_workers: int = 4,
        pin_memory: bool = True,
        drop_last: bool = False,
        label_extractor: Optional[LabelExtractor] = None,
        transform: Optional[Any] = None,
        target_transform: Optional[Callable] = None,
        device: Optional[str] = None,
        prefetch_factor: int = 2,
        # TurboLoader-specific
        enable_cache: bool = False,
        cache_l1_mb: int = 512,
        output_size: Optional[Tuple[int, int]] = None,
    ):
        """
        Args:
            data_path: Path to TAR/TBL file
            batch_size: Samples per batch
            shuffle: Whether to shuffle data
            num_workers: Number of worker threads
            pin_memory: Ignored (TurboLoader handles this internally)
            drop_last: Drop the last incomplete batch
            label_extractor: LabelExtractor instance for extracting labels
            transform: TurboLoader transform or Compose object
            target_transform: Optional transform for labels
            device: Target device ('cuda', 'cpu', or None for CPU)
            prefetch_factor: Batches to prefetch per worker
            enable_cache: Enable TurboLoader caching
            cache_l1_mb: L1 cache size in MB
            output_size: Optional (height, width) for resizing
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for PyTorchCompatibleLoader")

        self._data_path = data_path
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._num_workers = num_workers
        self._drop_last = drop_last
        self._transform = transform
        self._target_transform = target_transform
        self._device = device
        self._output_size = output_size

        # Default to folder-based label extraction
        self._label_extractor = label_extractor or FolderLabelExtractor()

        # Create underlying TurboLoader
        self._loader = turboloader.FastDataLoader(
            data_path,
            batch_size=batch_size,
            num_workers=num_workers,
            output_format="pytorch",
            target_height=output_size[0] if output_size else 224,
            target_width=output_size[1] if output_size else 224,
            transform=transform,
            shuffle=shuffle,
            drop_last=drop_last,
            enable_cache=enable_cache,
            cache_l1_mb=cache_l1_mb,
            prefetch_batches=prefetch_factor * num_workers,
        )

        self._epoch = 0
        self._iterator = None

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Iterate over batches, yielding (images, labels) tuples."""
        self._iterator = self._create_iterator()
        return self

    def _create_iterator(self):
        """Create a new iterator over the data."""
        # Reset loader for new epoch
        try:
            self._loader.stop()
        except:
            pass

        self._loader = turboloader.FastDataLoader(
            self._data_path,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            output_format="pytorch",
            target_height=self._output_size[0] if self._output_size else 224,
            target_width=self._output_size[1] if self._output_size else 224,
            transform=self._transform,
            shuffle=self._shuffle,
            drop_last=self._drop_last,
        )

        while True:
            try:
                images, metadata = self._loader.next_batch_torch(device=self._device)

                if images.numel() == 0:
                    if self._loader.is_finished():
                        break
                    continue

                # Extract labels
                filenames = metadata.get("filenames", [])
                labels = []

                for i, fn in enumerate(filenames):
                    sample_meta = {
                        k: v[i] if isinstance(v, list) else v for k, v in metadata.items()
                    }
                    label = self._label_extractor.extract(fn, sample_meta)
                    labels.append(label)

                labels_tensor = torch.tensor(labels, dtype=torch.long)

                if self._device:
                    labels_tensor = labels_tensor.to(self._device)

                if self._target_transform:
                    labels_tensor = self._target_transform(labels_tensor)

                yield images, labels_tensor

            except StopIteration:
                break

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get next batch."""
        if self._iterator is None:
            self._iterator = self._create_iterator()
        return next(self._iterator)

    def __len__(self) -> int:
        """Approximate length (number of batches)."""
        # This is an estimate; TurboLoader doesn't expose exact length
        return 0  # Unknown length

    def set_epoch(self, epoch: int):
        """Set epoch for reproducible shuffling."""
        self._epoch = epoch

    @property
    def dataset(self):
        """Compatibility property - returns self."""
        return self

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def num_workers(self) -> int:
        return self._num_workers

    @property
    def label_extractor(self) -> LabelExtractor:
        return self._label_extractor

    def close(self):
        """Clean up resources."""
        try:
            self._loader.stop()
        except:
            pass

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# =============================================================================
# IMAGE FOLDER CONVERTER
# =============================================================================


class ImageFolderConverter:
    """Convert ImageFolder-style directories to TurboLoader TAR format.

    Handles the common ImageFolder structure:
        root/
            class1/
                img1.jpg
                img2.jpg
            class2/
                img3.jpg
                ...

    Example:
        converter = ImageFolderConverter()
        class_to_idx = converter.convert(
            '/path/to/imagenet/train',
            'imagenet_train.tar',
            show_progress=True
        )
    """

    VALID_EXTENSIONS = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".gif",
        ".webp",
        ".tiff",
        ".tif",
        ".ppm",
        ".pgm",
        ".pbm",
    }

    def __init__(self, extensions: Optional[set] = None):
        """
        Args:
            extensions: Set of valid file extensions (with dot).
                       Defaults to common image formats.
        """
        self._extensions = extensions or self.VALID_EXTENSIONS

    def convert(
        self,
        source_dir: str,
        output_path: str,
        show_progress: bool = True,
        compression: Optional[str] = None,
        save_class_mapping: bool = True,
    ) -> Dict[str, int]:
        """Convert ImageFolder directory to TAR file.

        Args:
            source_dir: Path to ImageFolder root directory
            output_path: Path for output TAR file
            show_progress: Show progress bar (requires tqdm)
            compression: TAR compression ('gz', 'bz2', 'xz', or None)
            save_class_mapping: Save class_to_idx.json alongside TAR

        Returns:
            Dict mapping class names to indices
        """
        source_path = Path(source_dir)
        if not source_path.is_dir():
            raise ValueError(f"Source directory not found: {source_dir}")

        # Discover classes
        classes = sorted(
            [d.name for d in source_path.iterdir() if d.is_dir() and not d.name.startswith(".")]
        )

        if not classes:
            raise ValueError(f"No class directories found in {source_dir}")

        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

        # Collect all image files
        all_files = []
        for class_name in classes:
            class_dir = source_path / class_name
            for ext in self._extensions:
                all_files.extend([(f, class_name) for f in class_dir.rglob(f"*{ext}")])
                # Also check uppercase
                all_files.extend([(f, class_name) for f in class_dir.rglob(f"*{ext.upper()}")])

        # Remove duplicates and sort
        all_files = sorted(set(all_files), key=lambda x: str(x[0]))

        if not all_files:
            raise ValueError(f"No image files found in {source_dir}")

        print(f"Found {len(all_files)} images in {len(classes)} classes")

        # Determine TAR mode
        if compression:
            mode = f"w:{compression}"
        else:
            mode = "w"

        # Create TAR file
        if show_progress:
            try:
                from tqdm import tqdm

                iterator = tqdm(all_files, desc="Converting", unit="img")
            except ImportError:
                iterator = all_files
                print("Install tqdm for progress bar: pip install tqdm")
        else:
            iterator = all_files

        with tarfile.open(output_path, mode) as tar:
            for file_path, class_name in iterator:
                # Use relative path within class folder
                rel_path = f"{class_name}/{file_path.name}"
                tar.add(str(file_path), arcname=rel_path)

        # Save class mapping
        if save_class_mapping:
            mapping_path = output_path.rsplit(".", 1)[0] + "_classes.json"
            with open(mapping_path, "w") as f:
                json.dump(
                    {
                        "class_to_idx": class_to_idx,
                        "classes": classes,
                        "num_images": len(all_files),
                    },
                    f,
                    indent=2,
                )
            print(f"Saved class mapping to: {mapping_path}")

        print(f"Created TAR file: {output_path}")
        print(f"  Classes: {len(classes)}")
        print(f"  Images: {len(all_files)}")

        return class_to_idx

    def convert_dataset(
        self,
        dataset,
        output_path: str,
        show_progress: bool = True,
    ) -> Dict[str, int]:
        """Convert a PyTorch Dataset to TAR file.

        Works with any dataset that returns (image, label) and has
        samples attribute (like ImageFolder).

        Args:
            dataset: PyTorch Dataset with samples attribute
            output_path: Path for output TAR file
            show_progress: Show progress bar

        Returns:
            Dict mapping class names to indices
        """
        from PIL import Image
        from io import BytesIO

        if not hasattr(dataset, "samples"):
            raise ValueError("Dataset must have 'samples' attribute (like ImageFolder)")

        if hasattr(dataset, "class_to_idx"):
            class_to_idx = dataset.class_to_idx
        else:
            class_to_idx = {}

        if hasattr(dataset, "classes"):
            classes = dataset.classes
        else:
            classes = list(class_to_idx.keys())

        samples = dataset.samples  # List of (path, label)

        if show_progress:
            try:
                from tqdm import tqdm

                iterator = tqdm(
                    enumerate(samples), total=len(samples), desc="Converting", unit="img"
                )
            except ImportError:
                iterator = enumerate(samples)

        with tarfile.open(output_path, "w") as tar:
            for idx, (path, label) in iterator:
                file_path = Path(path)
                class_name = classes[label] if classes else f"class_{label}"
                rel_path = f"{class_name}/{file_path.name}"
                tar.add(str(file_path), arcname=rel_path)

        # Save class mapping
        mapping_path = output_path.rsplit(".", 1)[0] + "_classes.json"
        with open(mapping_path, "w") as f:
            json.dump(
                {
                    "class_to_idx": class_to_idx,
                    "classes": classes,
                    "num_images": len(samples),
                },
                f,
                indent=2,
            )

        print(f"Created TAR file: {output_path}")
        return class_to_idx


# =============================================================================
# TRANSFORM COMPATIBILITY
# =============================================================================


class TransformAdapter:
    """Adapt torchvision transforms to work with TurboLoader.

    Provides equivalent TurboLoader transforms for common torchvision ops.
    """

    # Mapping from torchvision transform names to TurboLoader equivalents
    TRANSFORM_MAP = {
        "Resize": "Resize",
        "CenterCrop": "CenterCrop",
        "RandomCrop": "RandomCrop",
        "RandomHorizontalFlip": "RandomHorizontalFlip",
        "RandomVerticalFlip": "RandomVerticalFlip",
        "ColorJitter": "ColorJitter",
        "Normalize": "Normalize",
        "ToTensor": "ToTensor",
        "RandomRotation": "RandomRotation",
        "GaussianBlur": "GaussianBlur",
        "Grayscale": "Grayscale",
        "Pad": "Pad",
        "RandomAffine": "RandomAffine",
        "RandomPerspective": "RandomPerspective",
        "RandomErasing": "RandomErasing",
    }

    @classmethod
    def from_torchvision(cls, tv_transforms) -> Any:
        """Convert torchvision Compose to TurboLoader Compose.

        Args:
            tv_transforms: torchvision.transforms.Compose object

        Returns:
            TurboLoader composed transform

        Example:
            tv_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            tl_transform = TransformAdapter.from_torchvision(tv_transform)
        """
        turbo_transforms = []

        for t in tv_transforms.transforms:
            name = type(t).__name__

            if name == "Resize":
                size = t.size
                if isinstance(size, int):
                    turbo_transforms.append(turboloader.Resize(size, size))
                else:
                    turbo_transforms.append(turboloader.Resize(size[0], size[1]))

            elif name == "CenterCrop":
                size = t.size
                if isinstance(size, int):
                    turbo_transforms.append(turboloader.CenterCrop(size, size))
                else:
                    turbo_transforms.append(turboloader.CenterCrop(size[0], size[1]))

            elif name == "RandomCrop":
                size = t.size
                if isinstance(size, int):
                    turbo_transforms.append(turboloader.RandomCrop(size, size))
                else:
                    turbo_transforms.append(turboloader.RandomCrop(size[0], size[1]))

            elif name == "RandomHorizontalFlip":
                p = getattr(t, "p", 0.5)
                turbo_transforms.append(turboloader.RandomHorizontalFlip(p))

            elif name == "RandomVerticalFlip":
                p = getattr(t, "p", 0.5)
                turbo_transforms.append(turboloader.RandomVerticalFlip(p))

            elif name == "ColorJitter":
                turbo_transforms.append(
                    turboloader.ColorJitter(
                        brightness=t.brightness or 0,
                        contrast=t.contrast or 0,
                        saturation=t.saturation or 0,
                        hue=t.hue or 0,
                    )
                )

            elif name == "Normalize":
                mean = list(t.mean) if hasattr(t.mean, "__iter__") else [t.mean]
                std = list(t.std) if hasattr(t.std, "__iter__") else [t.std]
                # Check for ImageNet normalization
                if (
                    abs(mean[0] - 0.485) < 0.01
                    and abs(mean[1] - 0.456) < 0.01
                    and abs(mean[2] - 0.406) < 0.01
                ):
                    turbo_transforms.append(turboloader.ImageNetNormalize())
                else:
                    turbo_transforms.append(turboloader.Normalize(mean, std))

            elif name == "ToTensor":
                turbo_transforms.append(turboloader.ToTensor())

            elif name == "GaussianBlur":
                kernel_size = t.kernel_size
                if isinstance(kernel_size, (list, tuple)):
                    kernel_size = kernel_size[0]
                sigma = t.sigma
                if isinstance(sigma, (list, tuple)):
                    sigma = sigma[0]
                turbo_transforms.append(turboloader.GaussianBlur(kernel_size, sigma))

            elif name == "Grayscale":
                turbo_transforms.append(turboloader.Grayscale())

            elif name == "RandomRotation":
                degrees = t.degrees
                if isinstance(degrees, (list, tuple)):
                    max_deg = max(abs(degrees[0]), abs(degrees[1]))
                else:
                    max_deg = degrees
                turbo_transforms.append(turboloader.RandomRotation(max_deg))

            elif name == "ToPILImage":
                # Skip - TurboLoader works with numpy arrays
                pass

            elif name == "Lambda":
                # Can't convert lambdas - skip with warning
                print(f"Warning: Cannot convert Lambda transform, skipping")

            else:
                print(f"Warning: Unknown transform '{name}', skipping")

        # Compose using pipe operator
        if not turbo_transforms:
            return None

        result = turbo_transforms[0]
        for t in turbo_transforms[1:]:
            result = result | t

        return result

    @classmethod
    def imagenet_train(cls):
        """Standard ImageNet training transforms."""
        return (
            turboloader.Resize(256, 256)
            | turboloader.RandomCrop(224, 224)
            | turboloader.RandomHorizontalFlip(0.5)
            | turboloader.ColorJitter(0.4, 0.4, 0.4, 0.1)
            | turboloader.ImageNetNormalize()
            | turboloader.ToTensor()
        )

    @classmethod
    def imagenet_val(cls):
        """Standard ImageNet validation transforms."""
        return (
            turboloader.Resize(256, 256)
            | turboloader.CenterCrop(224, 224)
            | turboloader.ImageNetNormalize()
            | turboloader.ToTensor()
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def create_loader(
    data_path: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    transform: str = "imagenet_train",
    device: Optional[str] = None,
    **kwargs,
) -> PyTorchCompatibleLoader:
    """Convenience function to create a PyTorch-compatible loader.

    Args:
        data_path: Path to TAR/TBL file
        batch_size: Batch size
        shuffle: Shuffle data
        num_workers: Number of workers
        transform: 'imagenet_train', 'imagenet_val', or TurboLoader transform
        device: Target device
        **kwargs: Additional arguments for PyTorchCompatibleLoader

    Returns:
        PyTorchCompatibleLoader instance
    """
    if transform == "imagenet_train":
        transform = TransformAdapter.imagenet_train()
    elif transform == "imagenet_val":
        transform = TransformAdapter.imagenet_val()

    return PyTorchCompatibleLoader(
        data_path,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        transform=transform,
        device=device,
        **kwargs,
    )


def convert_imagefolder(source_dir: str, output_tar: str, **kwargs) -> Dict[str, int]:
    """Convenience function to convert ImageFolder to TAR.

    Args:
        source_dir: ImageFolder directory
        output_tar: Output TAR path
        **kwargs: Additional arguments for ImageFolderConverter.convert

    Returns:
        class_to_idx mapping
    """
    converter = ImageFolderConverter()
    return converter.convert(source_dir, output_tar, **kwargs)


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Main classes
    "PyTorchCompatibleLoader",
    "ImageFolderConverter",
    "TransformAdapter",
    # Label extractors
    "LabelExtractor",
    "FolderLabelExtractor",
    "FilenamePatternExtractor",
    "MetadataLabelExtractor",
    "JSONSidecarExtractor",
    "CallableLabelExtractor",
    # Convenience functions
    "create_loader",
    "convert_imagefolder",
]
