"""
WebDataset integration for TurboLoader

Provides WebDataset-compatible interface using TurboLoader backend.
WebDataset format: TAR archives with files grouped by sample index
(e.g., 000000.jpg, 000000.json, 000001.jpg, 000001.json)
"""

import re
from typing import Optional, Callable, Iterator, Dict, Any
from collections import defaultdict

try:
    from _turboloader import DataLoader as _CppDataLoader
except ImportError:
    raise ImportError("TurboLoader C++ extension not found. "
                      "Make sure TurboLoader is properly installed.")


class WebDatasetLoader:
    """
    WebDataset-compatible DataLoader using TurboLoader backend.

    Automatically groups files from TAR archives by their base name
    (e.g., 000000.jpg and 000000.json become one sample).

    Example:
        >>> loader = WebDatasetLoader('dataset.tar', batch_size=32, num_workers=8)
        >>> for batch in loader:
        >>>     images = batch['jpg']  # or batch['png']
        >>>     labels = batch['json']
    """

    def __init__(
        self,
        data_path: str,
        batch_size: int = 32,
        num_workers: int = 4,
        shuffle: bool = False,
        transform: Optional[Callable] = None,
    ):
        """
        Initialize WebDataset loader.

        Args:
            data_path: Path to WebDataset TAR file
            batch_size: Samples per batch
            num_workers: Worker threads
            shuffle: Shuffle samples
            transform: Optional transform function (dict -> dict)
        """
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.transform = transform

        # Create C++ backend loader
        self._cpp_loader = _CppDataLoader(
            data_path=data_path,
            batch_size=batch_size * 10,  # Load more files, will group them
            num_workers=num_workers,
            shuffle=shuffle
        )

        # Pattern to extract base name and extension
        self._pattern = re.compile(r'^(.+?)\.([^.]+)$')

    def _group_by_sample(self, files: list) -> list:
        """
        Group files by their base name (sample ID).

        Example:
            ['000000.jpg', '000000.json', '000001.jpg', '000001.json']
            -> [{'jpg': ..., 'json': ...}, {'jpg': ..., 'json': ...}]
        """
        samples = defaultdict(dict)

        for file_data in files:
            filename = file_data.get('filename', '')
            match = self._pattern.match(filename)

            if match:
                base_name, extension = match.groups()

                # Store data under extension key
                samples[base_name][extension] = file_data.get('image')
                samples[base_name]['filename'] = filename

        # Convert to list of dicts
        return [sample for sample in samples.values()]

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over batches"""
        current_batch = []

        while not self._cpp_loader.is_finished():
            files = self._cpp_loader.next_batch()

            if not files:
                break

            # Group files by sample
            samples = self._group_by_sample(files)

            for sample in samples:
                # Apply transform if provided
                if self.transform is not None:
                    sample = self.transform(sample)

                current_batch.append(sample)

                # Yield batch when full
                if len(current_batch) >= self.batch_size:
                    yield current_batch[:self.batch_size]
                    current_batch = current_batch[self.batch_size:]

        # Yield remaining samples
        if current_batch:
            yield current_batch

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cpp_loader.stop()


def webdataset_decoder(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Default decoder for WebDataset samples.

    Converts image data to appropriate format based on extension.

    Args:
        sample: Dict with extension keys (e.g., 'jpg', 'json')

    Returns:
        Decoded sample dict
    """
    decoded = {}

    for key, value in sample.items():
        if key in ['jpg', 'jpeg', 'png', 'webp']:
            # Image data already decoded by TurboLoader
            decoded['image'] = value
        elif key == 'json':
            # Parse JSON if needed
            import json
            if isinstance(value, bytes):
                decoded['label'] = json.loads(value.decode('utf-8'))
            else:
                decoded['label'] = value
        elif key == 'txt':
            # Decode text
            if isinstance(value, bytes):
                decoded['text'] = value.decode('utf-8')
            else:
                decoded['text'] = value
        else:
            # Keep as-is
            decoded[key] = value

    return decoded


class PyTorchWebDataset:
    """
    PyTorch-compatible WebDataset using TurboLoader backend.

    Drop-in replacement for webdataset.WebDataset with TurboLoader performance.

    Example:
        >>> dataset = PyTorchWebDataset('data.tar')
        >>> loader = torch.utils.data.DataLoader(dataset, batch_size=None)
        >>> for batch in loader:
        >>>     images = batch['image']
        >>>     labels = batch['label']
    """

    def __init__(
        self,
        data_path: str,
        batch_size: int = 32,
        num_workers: int = 4,
        shuffle: bool = False,
        decode: bool = True,
        transform: Optional[Callable] = None,
    ):
        """
        Initialize PyTorch WebDataset.

        Args:
            data_path: Path to WebDataset TAR
            batch_size: Samples per batch
            num_workers: Worker threads
            shuffle: Shuffle samples
            decode: Automatically decode images/JSON (default: True)
            transform: Optional transform function
        """
        decoder = webdataset_decoder if decode else None

        self._loader = WebDatasetLoader(
            data_path=data_path,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            transform=decoder if not transform else lambda x: transform(decoder(x))
        )

    def __iter__(self):
        return iter(self._loader)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._loader.__exit__(exc_type, exc_val, exc_tb)
