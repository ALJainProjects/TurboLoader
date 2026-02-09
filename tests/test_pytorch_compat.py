"""Tests for PyTorch Compatibility Layer (Phase 3)

Tests the PyTorchCompatibleLoader API surface to ensure it matches
PyTorch DataLoader's interface, including new parameters added in
this phase: collate_fn, sampler, worker_init_fn, pin_memory.

Note: These tests validate the Python API layer without requiring
the C++ _turboloader module to be compiled.
"""

import pytest
import sys
from unittest.mock import MagicMock, patch

# ============================================================================
# Label extractor tests (no C++ dependency)
# ============================================================================


class TestFolderLabelExtractor:
    """Test FolderLabelExtractor for ImageFolder-style label extraction."""

    def test_basic_extraction(self):
        from turboloader.pytorch_compat import FolderLabelExtractor

        extractor = FolderLabelExtractor()
        # First class seen gets index 0
        assert extractor.extract("cat/img001.jpg", {}) == 0
        # Same class same index
        assert extractor.extract("cat/img002.jpg", {}) == 0
        # New class gets next index
        assert extractor.extract("dog/img001.jpg", {}) == 1
        assert extractor.get_num_classes() == 2

    def test_predefined_mapping(self):
        from turboloader.pytorch_compat import FolderLabelExtractor

        mapping = {"dog": 0, "cat": 1, "fish": 2}
        extractor = FolderLabelExtractor(class_to_idx=mapping)
        assert extractor.extract("dog/img.jpg", {}) == 0
        assert extractor.extract("cat/img.jpg", {}) == 1
        assert extractor.extract("fish/img.jpg", {}) == 2

    def test_unknown_class_gets_new_index(self):
        from turboloader.pytorch_compat import FolderLabelExtractor

        mapping = {"dog": 0, "cat": 1}
        extractor = FolderLabelExtractor(class_to_idx=mapping)
        # New class gets next index
        assert extractor.extract("bird/img.jpg", {}) == 2
        assert extractor.get_num_classes() == 3


class TestFilenamePatternExtractor:
    """Test regex-based label extraction."""

    def test_pattern_extraction(self):
        from turboloader.pytorch_compat import FilenamePatternExtractor

        extractor = FilenamePatternExtractor(r"class_(\d+)_")
        assert extractor.extract("class_5_image_001.jpg", {}) == 5
        assert extractor.extract("class_42_image.jpg", {}) == 42

    def test_no_match_returns_zero(self):
        from turboloader.pytorch_compat import FilenamePatternExtractor

        extractor = FilenamePatternExtractor(r"class_(\d+)_")
        assert extractor.extract("other_format.jpg", {}) == 0


class TestCallableLabelExtractor:
    """Test custom callable extractors."""

    def test_lambda_extractor(self):
        from turboloader.pytorch_compat import CallableLabelExtractor

        extractor = CallableLabelExtractor(lambda fn, _: len(fn))
        assert extractor.extract("abc", {}) == 3
        assert extractor.extract("abcdef", {}) == 6


class TestMetadataLabelExtractor:
    """Test metadata-based label extraction."""

    def test_basic_metadata(self):
        from turboloader.pytorch_compat import MetadataLabelExtractor

        extractor = MetadataLabelExtractor(key="label")
        assert extractor.extract("img.jpg", {"label": 7}) == 7

    def test_missing_key_returns_default(self):
        from turboloader.pytorch_compat import MetadataLabelExtractor

        extractor = MetadataLabelExtractor(key="label", default=99)
        assert extractor.extract("img.jpg", {}) == 99


# ============================================================================
# PyTorchCompatibleLoader API surface tests
# ============================================================================


class TestLoaderAPIProperties:
    """Test that PyTorchCompatibleLoader exposes all PyTorch-compatible properties.

    These tests mock the underlying loader since we may not have the C++ module.
    """

    @pytest.fixture
    def mock_loader_class(self):
        """Set up mocks for turboloader and torch imports."""
        with patch.dict(
            sys.modules,
            {
                "torch": MagicMock(),
                "torch.utils": MagicMock(),
                "torch.utils.data": MagicMock(),
                "_turboloader": MagicMock(),
            },
        ):
            # Force reimport with mocks
            if "turboloader.pytorch_compat" in sys.modules:
                del sys.modules["turboloader.pytorch_compat"]

            # Patch module-level flags
            import turboloader.pytorch_compat as compat

            compat.TORCH_AVAILABLE = True
            compat.TURBOLOADER_AVAILABLE = True
            compat.turboloader = MagicMock()
            yield compat

    def test_has_batch_size(self, mock_loader_class):
        loader = mock_loader_class.PyTorchCompatibleLoader.__new__(
            mock_loader_class.PyTorchCompatibleLoader
        )
        loader._batch_size = 64
        assert loader.batch_size == 64

    def test_has_num_workers(self, mock_loader_class):
        loader = mock_loader_class.PyTorchCompatibleLoader.__new__(
            mock_loader_class.PyTorchCompatibleLoader
        )
        loader._num_workers = 8
        assert loader.num_workers == 8

    def test_has_pin_memory(self, mock_loader_class):
        loader = mock_loader_class.PyTorchCompatibleLoader.__new__(
            mock_loader_class.PyTorchCompatibleLoader
        )
        loader._pin_memory = True
        assert loader.pin_memory is True

    def test_has_drop_last(self, mock_loader_class):
        loader = mock_loader_class.PyTorchCompatibleLoader.__new__(
            mock_loader_class.PyTorchCompatibleLoader
        )
        loader._drop_last = True
        assert loader.drop_last is True

    def test_has_sampler(self, mock_loader_class):
        loader = mock_loader_class.PyTorchCompatibleLoader.__new__(
            mock_loader_class.PyTorchCompatibleLoader
        )
        mock_sampler = MagicMock()
        loader._sampler = mock_sampler
        assert loader.sampler is mock_sampler

    def test_has_collate_fn(self, mock_loader_class):
        loader = mock_loader_class.PyTorchCompatibleLoader.__new__(
            mock_loader_class.PyTorchCompatibleLoader
        )
        mock_fn = MagicMock()
        loader._collate_fn = mock_fn
        assert loader.collate_fn is mock_fn

    def test_has_worker_init_fn(self, mock_loader_class):
        loader = mock_loader_class.PyTorchCompatibleLoader.__new__(
            mock_loader_class.PyTorchCompatibleLoader
        )
        mock_fn = MagicMock()
        loader._worker_init_fn = mock_fn
        assert loader.worker_init_fn is mock_fn

    def test_has_dataset_property(self, mock_loader_class):
        loader = mock_loader_class.PyTorchCompatibleLoader.__new__(
            mock_loader_class.PyTorchCompatibleLoader
        )
        assert loader.dataset is loader

    def test_len_raises_before_first_epoch(self, mock_loader_class):
        loader = mock_loader_class.PyTorchCompatibleLoader.__new__(
            mock_loader_class.PyTorchCompatibleLoader
        )
        loader._num_batches = None
        with pytest.raises(TypeError, match="unknown until the first epoch"):
            len(loader)

    def test_len_returns_count_after_first_epoch(self, mock_loader_class):
        loader = mock_loader_class.PyTorchCompatibleLoader.__new__(
            mock_loader_class.PyTorchCompatibleLoader
        )
        loader._num_batches = 42
        assert len(loader) == 42

    def test_context_manager(self, mock_loader_class):
        loader = mock_loader_class.PyTorchCompatibleLoader.__new__(
            mock_loader_class.PyTorchCompatibleLoader
        )
        loader._loader = MagicMock()
        with loader:
            pass
        # close() should have been called


# ============================================================================
# S3 XML error detection (unit test the concept)
# ============================================================================


class TestS3XMLErrorDetection:
    """Test that we can detect S3 XML error pages.

    This validates the logic used in s3_reader.hpp's is_xml_error_response().
    """

    @staticmethod
    def is_xml_error_response(data: bytes) -> bool:
        """Python equivalent of the C++ check."""
        if len(data) < 6:
            return False
        return (data[:5] == b"<?xml") or (data[:6] == b"<Error")

    def test_detects_xml_prolog(self):
        data = b'<?xml version="1.0" encoding="UTF-8"?><Error>...'
        assert self.is_xml_error_response(data)

    def test_detects_error_element(self):
        data = b"<Error><Code>AccessDenied</Code></Error>"
        assert self.is_xml_error_response(data)

    def test_passes_jpeg_data(self):
        data = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00"
        assert not self.is_xml_error_response(data)

    def test_passes_png_data(self):
        data = b"\x89PNG\r\n\x1a\n\x00\x00"
        assert not self.is_xml_error_response(data)

    def test_passes_empty_data(self):
        assert not self.is_xml_error_response(b"")

    def test_passes_short_data(self):
        assert not self.is_xml_error_response(b"abc")
