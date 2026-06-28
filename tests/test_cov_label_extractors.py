"""Coverage/regression tests for the label-extractor classes in
``turboloader.pytorch_compat``.

Targets the following public API (see ``__all__`` of pytorch_compat.py):

    - LabelExtractor          (abstract base class)
    - FolderLabelExtractor
    - FilenamePatternExtractor
    - MetadataLabelExtractor
    - JSONSidecarExtractor
    - CallableLabelExtractor

The label extractors do NOT depend on torch/torchvision (torch is an optional
dependency in the module, guarded by try/except), so we import the classes
directly. ``pytest.importorskip`` is only used defensively in case the
compiled ``turboloader`` / numpy stack is unavailable in some environment.
"""

import json

import pytest

# numpy is a hard dependency of pytorch_compat (imported at module top-level).
pytest.importorskip("numpy")

pc = pytest.importorskip("turboloader.pytorch_compat")

from turboloader.pytorch_compat import (  # noqa: E402
    LabelExtractor,
    FolderLabelExtractor,
    FilenamePatternExtractor,
    MetadataLabelExtractor,
    JSONSidecarExtractor,
    CallableLabelExtractor,
)


# ---------------------------------------------------------------------------
# LabelExtractor (abstract base class)
# ---------------------------------------------------------------------------


def test_labelextractor_is_abstract_cannot_instantiate():
    """The ABC declares ``extract`` as abstract, so direct construction must
    raise TypeError."""
    with pytest.raises(TypeError):
        LabelExtractor()


def test_labelextractor_subclass_without_extract_is_abstract():
    """A subclass that fails to implement ``extract`` is still abstract."""

    class Incomplete(LabelExtractor):
        pass

    with pytest.raises(TypeError):
        Incomplete()


def test_labelextractor_base_defaults_return_none():
    """``get_num_classes`` / ``get_class_names`` default to None on the base
    class. Verify via a minimal concrete subclass."""

    class Minimal(LabelExtractor):
        def extract(self, filename, metadata):
            return 0

    m = Minimal()
    assert m.extract("anything", {}) == 0
    assert m.get_num_classes() is None
    assert m.get_class_names() is None


def test_all_extractors_are_labelextractor_subclasses():
    for cls in (
        FolderLabelExtractor,
        FilenamePatternExtractor,
        MetadataLabelExtractor,
        JSONSidecarExtractor,
        CallableLabelExtractor,
    ):
        assert issubclass(cls, LabelExtractor)


# ---------------------------------------------------------------------------
# FolderLabelExtractor
# ---------------------------------------------------------------------------


def test_folder_assigns_indices_in_first_seen_order():
    ex = FolderLabelExtractor()
    # Indices are assigned on first encounter, NOT sorted alphabetically.
    assert ex.extract("dog/img1.jpg", {}) == 0
    assert ex.extract("cat/img1.jpg", {}) == 1
    # Re-seeing a class returns the stable, already-assigned index.
    assert ex.extract("dog/img2.jpg", {}) == 0
    assert ex.extract("cat/img9.jpg", {}) == 1
    # A third class gets the next free index.
    assert ex.extract("fish/x.jpg", {}) == 2


def test_folder_uses_first_path_component_as_class():
    ex = FolderLabelExtractor()
    # Nested subdirectories: only the top folder is the class.
    assert ex.extract("cat/sub/deep/img.jpg", {}) == 0
    assert ex.extract("cat/other/img.jpg", {}) == 0
    assert ex.get_num_classes() == 1


def test_folder_normalizes_windows_backslashes():
    ex = FolderLabelExtractor()
    assert ex.extract("dog\\sub\\img.jpg", {}) == 0
    # Same class reached via forward slashes must map to the same index.
    assert ex.extract("dog/another.jpg", {}) == 0
    assert ex.get_num_classes() == 1


def test_folder_no_separator_maps_to_unknown():
    ex = FolderLabelExtractor()
    assert ex.extract("loose_image.jpg", {}) == 0
    # Any other separator-less name collapses to the same "unknown" class.
    assert ex.extract("another_loose.png", {}) == 0
    assert ex.class_to_idx == {"unknown": 0}


def test_folder_num_classes_and_class_names_track_dynamic_state():
    ex = FolderLabelExtractor()
    ex.extract("b/x.jpg", {})  # -> 0
    ex.extract("a/x.jpg", {})  # -> 1
    ex.extract("c/x.jpg", {})  # -> 2
    assert ex.get_num_classes() == 3
    # get_class_names returns names ordered by their assigned index.
    assert ex.get_class_names() == ["b", "a", "c"]


def test_folder_prebuilt_mapping_is_respected_and_extended():
    ex = FolderLabelExtractor(class_to_idx={"cat": 0, "dog": 1})
    assert ex.extract("cat/x.jpg", {}) == 0
    assert ex.extract("dog/x.jpg", {}) == 1
    # New class continues from max(existing)+1.
    assert ex.extract("fish/x.jpg", {}) == 2
    assert ex.get_num_classes() == 3
    assert ex.get_class_names() == ["cat", "dog", "fish"]


def test_folder_class_to_idx_property_returns_copy():
    ex = FolderLabelExtractor()
    ex.extract("cat/x.jpg", {})
    snapshot = ex.class_to_idx
    assert snapshot == {"cat": 0}
    # Mutating the returned dict must not corrupt internal state.
    snapshot["hacked"] = 99
    assert ex.class_to_idx == {"cat": 0}
    assert "hacked" not in ex.class_to_idx


def test_folder_metadata_is_ignored():
    """FolderLabelExtractor derives the label purely from the path; metadata
    contents are irrelevant."""
    ex = FolderLabelExtractor()
    assert ex.extract("cat/x.jpg", {"label": 123}) == 0


def test_folder_noncontiguous_prebuilt_mapping_class_names():
    # Regression: a prebuilt non-contiguous mapping must not crash get_class_names()
    # (it used to KeyError(0) by iterating range(len)). Now ordered by index value.
    ex = FolderLabelExtractor(class_to_idx={"a": 2, "b": 5})
    assert ex.get_num_classes() == 2
    assert ex.get_class_names() == ["a", "b"]


# ---------------------------------------------------------------------------
# FilenamePatternExtractor
# ---------------------------------------------------------------------------


def test_pattern_extracts_capture_group_as_int():
    ex = FilenamePatternExtractor(r"class_(\d+)_.*\.jpg")
    assert ex.extract("class_5_image_001.jpg", {}) == 5
    assert ex.extract("class_17_foo.jpg", {}) == 17


def test_pattern_no_match_returns_zero():
    ex = FilenamePatternExtractor(r"class_(\d+)_.*\.jpg")
    assert ex.extract("totally_unrelated.png", {}) == 0


def test_pattern_uses_search_not_fullmatch():
    """``search`` means the pattern can match anywhere in the string."""
    ex = FilenamePatternExtractor(r"(\d+)")
    assert ex.extract("prefix/dir/sample0042suffix.bin", {}) == 42


def test_pattern_respects_custom_group_index():
    ex = FilenamePatternExtractor(r"(\d+)_(\d+)", group=2)
    assert ex.extract("12_34_image.jpg", {}) == 34


def test_pattern_nonnumeric_capture_raises_value_error():
    """Capturing non-digits and feeding them to int() surfaces a ValueError —
    documenting the real (un-guarded) behavior."""
    ex = FilenamePatternExtractor(r"([a-z]+)")
    with pytest.raises(ValueError):
        ex.extract("abc", {})


def test_pattern_metadata_ignored():
    ex = FilenamePatternExtractor(r"_(\d+)\.")
    assert ex.extract("img_8.jpg", {"label": 999}) == 8


# ---------------------------------------------------------------------------
# MetadataLabelExtractor
# ---------------------------------------------------------------------------


def test_metadata_reads_default_label_key():
    ex = MetadataLabelExtractor()
    assert ex.extract("anything.jpg", {"label": 7}) == 7


def test_metadata_missing_key_returns_default_zero():
    ex = MetadataLabelExtractor()
    assert ex.extract("anything.jpg", {}) == 0


def test_metadata_custom_key_and_default():
    ex = MetadataLabelExtractor(key="category", default=-1)
    assert ex.extract("x.jpg", {"category": 3}) == 3
    assert ex.extract("x.jpg", {}) == -1
    # The default-key "label" is not consulted when a custom key is set.
    assert ex.extract("x.jpg", {"label": 99}) == -1


def test_metadata_filename_ignored():
    ex = MetadataLabelExtractor()
    assert ex.extract("dog/x.jpg", {"label": 4}) == 4


# ---------------------------------------------------------------------------
# JSONSidecarExtractor
# ---------------------------------------------------------------------------


def test_json_reads_label_from_json_data_in_metadata():
    ex = JSONSidecarExtractor()
    assert ex.extract("a.jpg", {"json_data": {"label": 9}}) == 9


def test_json_missing_json_data_returns_zero():
    ex = JSONSidecarExtractor()
    assert ex.extract("b.jpg", {}) == 0


def test_json_custom_label_key():
    ex = JSONSidecarExtractor(label_key="category")
    assert ex.extract("c.jpg", {"json_data": {"category": 4}}) == 4
    # Wrong key inside json_data falls back to 0.
    assert ex.extract("d.jpg", {"json_data": {"label": 4}}) == 0


def test_json_cache_returns_first_seen_value_for_same_filename():
    ex = JSONSidecarExtractor(cache=True)
    assert ex.extract("same.jpg", {"json_data": {"label": 1}}) == 1
    # With caching on, the filename keys the cache: subsequent metadata for the
    # same filename is ignored and the first value is returned.
    assert ex.extract("same.jpg", {"json_data": {"label": 999}}) == 1


def test_json_cache_disabled_recomputes_each_call():
    ex = JSONSidecarExtractor(cache=False)
    assert ex.extract("same.jpg", {"json_data": {"label": 1}}) == 1
    assert ex.extract("same.jpg", {"json_data": {"label": 999}}) == 999


def test_json_different_filenames_cached_independently():
    ex = JSONSidecarExtractor(cache=True)
    assert ex.extract("one.jpg", {"json_data": {"label": 10}}) == 10
    assert ex.extract("two.jpg", {"json_data": {"label": 20}}) == 20
    # Re-querying returns each filename's own cached value.
    assert ex.extract("one.jpg", {"json_data": {"label": -1}}) == 10
    assert ex.extract("two.jpg", {"json_data": {"label": -1}}) == 20


def test_json_does_not_read_real_sidecar_file_on_disk(tmp_path):
    """Despite the docstring mentioning ".json sidecar files", the
    implementation ONLY consults ``metadata['json_data']`` and never opens any
    file. A real sidecar present on disk is therefore ignored.

    This asserts the actual behavior (returns the default 0) using a real file
    created under tmp_path.
    """
    img = tmp_path / "image.jpg"
    img.write_bytes(b"not-a-real-jpeg")
    sidecar = tmp_path / "image.json"
    sidecar.write_text(json.dumps({"label": 42}))

    ex = JSONSidecarExtractor()
    # Pass the on-disk path but NO json_data in metadata -> file is not read.
    assert ex.extract(str(img), {}) == 0


# ---------------------------------------------------------------------------
# CallableLabelExtractor
# ---------------------------------------------------------------------------


def test_callable_uses_filename():
    ex = CallableLabelExtractor(lambda fn, meta: int(fn.split("_")[1]))
    assert ex.extract("img_42_x.jpg", {}) == 42


def test_callable_uses_metadata():
    ex = CallableLabelExtractor(lambda fn, meta: meta.get("y", 0) + 1)
    assert ex.extract("whatever", {"y": 10}) == 11
    assert ex.extract("whatever", {}) == 1


def test_callable_receives_both_arguments_in_order():
    seen = {}

    def fn(filename, metadata):
        seen["filename"] = filename
        seen["metadata"] = metadata
        return 5

    ex = CallableLabelExtractor(fn)
    result = ex.extract("dog/x.jpg", {"k": "v"})
    assert result == 5
    assert seen == {"filename": "dog/x.jpg", "metadata": {"k": "v"}}


def test_callable_propagates_exceptions():
    def boom(fn, meta):
        raise RuntimeError("callable failed")

    ex = CallableLabelExtractor(boom)
    with pytest.raises(RuntimeError, match="callable failed"):
        ex.extract("x", {})


def test_callable_default_num_classes_and_names_are_none():
    ex = CallableLabelExtractor(lambda fn, meta: 0)
    assert ex.get_num_classes() is None
    assert ex.get_class_names() is None
