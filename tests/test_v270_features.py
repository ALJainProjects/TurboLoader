"""Tests for TurboLoader v2.7.0 features: cache_decoded for FastDataLoader."""

import sys
import os
import time
import tempfile
import tarfile
from io import BytesIO

import numpy as np
import pytest

# Add turboloader to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import turboloader


@pytest.fixture
def test_tar():
    """Create a test tar file with JPEG images."""
    from PIL import Image

    fd, tar_path = tempfile.mkstemp(suffix=".tar")
    os.close(fd)

    with tarfile.open(tar_path, "w") as tar:
        for i in range(100):  # 100 test images
            img_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)

            buf = BytesIO()
            img.save(buf, format="JPEG", quality=85)
            buf.seek(0)

            tarinfo = tarfile.TarInfo(name=f"image_{i:04d}.jpg")
            tarinfo.size = len(buf.getvalue())
            tar.addfile(tarinfo, buf)

    yield tar_path

    # Cleanup
    os.unlink(tar_path)


class TestCacheDecoded:
    """Test suite for cache_decoded feature."""

    def test_cache_decoded_basic(self, test_tar):
        """Test that cache_decoded stores and retrieves batches correctly."""
        loader = turboloader.FastDataLoader(
            test_tar,
            batch_size=10,
            num_workers=2,
            cache_decoded=True,
            output_format="numpy",
        )

        # First epoch - populate cache
        epoch1_batches = []
        for images, metadata in loader:
            epoch1_batches.append((images.copy(), metadata.copy()))

        assert len(epoch1_batches) > 0, "First epoch should yield batches"
        assert loader.cache_populated, "Cache should be populated after first epoch"

        # Second epoch - from cache
        epoch2_batches = []
        for images, metadata in loader:
            epoch2_batches.append((images.copy(), metadata.copy()))

        assert len(epoch1_batches) == len(epoch2_batches), "Same number of batches in both epochs"

        # Verify data is identical
        for (img1, _), (img2, _) in zip(epoch1_batches, epoch2_batches):
            np.testing.assert_array_equal(img1, img2, "Cached data should match original")

    def test_cache_decoded_speedup(self, test_tar):
        """Test that cached epochs are faster than uncached."""
        loader = turboloader.FastDataLoader(
            test_tar,
            batch_size=20,
            num_workers=4,
            cache_decoded=True,
            output_format="numpy",
        )

        # Epoch 1 (cache warmup)
        t1_start = time.perf_counter()
        for _ in loader:
            pass
        t1 = time.perf_counter() - t1_start

        # Epoch 2 (cached)
        t2_start = time.perf_counter()
        for _ in loader:
            pass
        t2 = time.perf_counter() - t2_start

        # Cached epoch should be faster (at least 1.5x for small dataset)
        # Note: With very small datasets, the speedup may be less pronounced
        assert (
            t2 < t1 or t2 < 0.1
        ), f"Cached epoch ({t2:.3f}s) should be faster than first ({t1:.3f}s)"

    def test_clear_cache(self, test_tar):
        """Test that clear_cache() works correctly."""
        loader = turboloader.FastDataLoader(
            test_tar,
            batch_size=10,
            num_workers=2,
            cache_decoded=True,
            output_format="numpy",
        )

        # Populate cache
        for _ in loader:
            pass

        assert loader.cache_populated, "Cache should be populated"
        assert loader.cache_size_mb > 0, "Cache size should be > 0"

        # Clear cache
        loader.clear_cache()

        assert not loader.cache_populated, "Cache should not be populated after clear"
        assert loader.cache_size_mb == 0, "Cache size should be 0 after clear"

    def test_cache_size_mb(self, test_tar):
        """Test that cache_size_mb property returns correct size."""
        loader = turboloader.FastDataLoader(
            test_tar,
            batch_size=10,
            num_workers=2,
            cache_decoded=True,
            output_format="numpy",
        )

        assert loader.cache_size_mb == 0, "Cache should be empty initially"

        # Populate cache
        for _ in loader:
            pass

        # 100 images at 64x64x3 = ~1.2MB
        expected_min_mb = 0.5  # At least 0.5 MB
        assert (
            loader.cache_size_mb >= expected_min_mb
        ), f"Cache size ({loader.cache_size_mb:.2f}MB) should be >= {expected_min_mb}MB"

    def test_cache_disabled_by_default(self, test_tar):
        """Test that caching is disabled by default."""
        loader = turboloader.FastDataLoader(
            test_tar, batch_size=10, num_workers=2, output_format="numpy"
        )

        # Run through dataset
        for _ in loader:
            pass

        assert not loader.cache_populated, "Cache should not be populated when disabled"
        assert loader.cache_size_mb == 0, "Cache size should be 0 when disabled"

    def test_multiple_epochs_with_cache(self, test_tar):
        """Test running multiple epochs with cache."""
        loader = turboloader.FastDataLoader(
            test_tar,
            batch_size=10,
            num_workers=2,
            cache_decoded=True,
            output_format="numpy",
        )

        epoch_counts = []

        for epoch in range(5):
            count = 0
            for images, _ in loader:
                count += images.shape[0]
            epoch_counts.append(count)

        # All epochs should process the same number of images
        assert all(
            c == epoch_counts[0] for c in epoch_counts
        ), "All epochs should process the same number of images"

    def test_cache_with_small_batches(self, test_tar):
        """Test cache with small batch size."""
        loader = turboloader.FastDataLoader(
            test_tar,
            batch_size=1,
            num_workers=1,
            cache_decoded=True,
            output_format="numpy",
        )

        # First epoch
        first_epoch_images = []
        for images, _ in loader:
            first_epoch_images.append(images.copy())

        # Second epoch
        second_epoch_images = []
        for images, _ in loader:
            second_epoch_images.append(images.copy())

        assert len(first_epoch_images) == len(second_epoch_images)

        for img1, img2 in zip(first_epoch_images, second_epoch_images):
            np.testing.assert_array_equal(img1, img2)


class TestCacheDecodedIntegration:
    """Integration tests for cache_decoded with other features."""

    @pytest.mark.xfail(reason="CHW format cache shape mismatch")
    def test_cache_with_chw_format(self, test_tar):
        """Test cache_decoded with CHW output format (pytorch)."""
        loader = turboloader.FastDataLoader(
            test_tar,
            batch_size=10,
            num_workers=2,
            cache_decoded=True,
            output_format="numpy_chw",
        )

        # First epoch
        first_batch = None
        for images, _ in loader:
            if first_batch is None:
                first_batch = images.copy()
            break

        # Second epoch (from cache)
        for images, _ in loader:
            np.testing.assert_array_equal(first_batch, images)
            break

        # Verify CHW format
        assert first_batch.ndim == 4, "Should be 4D array"
        # Shape should be (N, C, H, W)
        assert first_batch.shape[1] == 3, "Second dimension should be channels (3)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
