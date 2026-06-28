"""Coverage / regression tests for turboloader.pytorch_compat.

These exercise the REAL pipeline end-to-end (no mocks): build a tiny
ImageFolder of JPEGs with PIL, convert it with ImageFolderConverter, and
drive PyTorchCompatibleLoader for actual (images, labels) batches. Also
covers TransformAdapter bridging from torchvision.

Everything is deterministic (fixed colors / sorted classes) and
self-contained (tmp_path / tmp_path_factory). Requires torch + torchvision
+ PIL + the compiled _turboloader extension; skipped otherwise.
"""

import json
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("PIL")

from PIL import Image

from turboloader.pytorch_compat import (
    ImageFolderConverter,
    PyTorchCompatibleLoader,
    FolderLabelExtractor,
    TransformAdapter,
)

# Sorted class names -> deterministic label indices 0,1,2
CLASSES = ["ant", "bee", "cat"]
PER_CLASS = 4
TOTAL = len(CLASSES) * PER_CLASS
# Distinct solid colors per class so each JPEG is a valid, decodable image.
COLORS = [(200, 30, 30), (30, 200, 30), (30, 30, 200)]


def _build_image_folder(root, src_h=40, src_w=40):
    """Write an ImageFolder tree of solid-color JPEGs under ``root``."""
    for ci, cls in enumerate(CLASSES):
        cdir = os.path.join(root, cls)
        os.makedirs(cdir, exist_ok=True)
        for i in range(PER_CLASS):
            arr = np.empty((src_h, src_w, 3), dtype=np.uint8)
            arr[:, :] = COLORS[ci]
            Image.fromarray(arr).save(os.path.join(cdir, f"{cls}_{i:03d}.jpg"), quality=95)
    return root


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    """Build the ImageFolder once and convert it to a TAR for the module."""
    base = tmp_path_factory.mktemp("imgfolder")
    root = _build_image_folder(str(base / "images"))
    tar_path = str(base / "data.tar")
    class_to_idx = ImageFolderConverter().convert(root, tar_path, show_progress=False)
    return {"root": root, "tar": tar_path, "class_to_idx": class_to_idx, "base": str(base)}


# ---------------------------------------------------------------------------
# ImageFolderConverter
# ---------------------------------------------------------------------------


def test_converter_returns_sorted_class_to_idx(dataset):
    # Classes are assigned indices by sorted folder name.
    assert dataset["class_to_idx"] == {"ant": 0, "bee": 1, "cat": 2}


def test_converter_creates_tar_and_sidecar_json(dataset):
    import tarfile

    tar_path = dataset["tar"]
    assert os.path.exists(tar_path)
    # The TAR holds one member per image, named "<class>/<file>".
    with tarfile.open(tar_path) as tf:
        names = sorted(tf.getnames())
    assert len(names) == TOTAL
    assert all("/" in n and n.split("/")[0] in CLASSES for n in names)

    # Sidecar mapping written next to the TAR: data.tar -> data_classes.json
    sidecar = tar_path.rsplit(".", 1)[0] + "_classes.json"
    assert os.path.exists(sidecar)
    with open(sidecar) as f:
        meta = json.load(f)
    assert meta["class_to_idx"] == {"ant": 0, "bee": 1, "cat": 2}
    assert meta["classes"] == CLASSES
    assert meta["num_images"] == TOTAL


def test_converter_save_class_mapping_false(tmp_path):
    root = _build_image_folder(str(tmp_path / "images"))
    out = str(tmp_path / "nomap.tar")
    ImageFolderConverter().convert(root, out, show_progress=False, save_class_mapping=False)
    assert os.path.exists(out)
    assert not os.path.exists(str(tmp_path / "nomap_classes.json"))


def test_converter_missing_source_raises(tmp_path):
    with pytest.raises(ValueError, match="Source directory not found"):
        ImageFolderConverter().convert(
            str(tmp_path / "does_not_exist"), str(tmp_path / "o.tar"), show_progress=False
        )


def test_converter_no_class_dirs_raises(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="No class directories"):
        ImageFolderConverter().convert(str(empty), str(tmp_path / "o.tar"), show_progress=False)


def test_converter_no_images_raises(tmp_path):
    root = tmp_path / "root"
    (root / "cls").mkdir(parents=True)
    (root / "cls" / "notes.txt").write_text("not an image")
    with pytest.raises(ValueError, match="No image files"):
        ImageFolderConverter().convert(str(root), str(tmp_path / "o.tar"), show_progress=False)


def test_converter_respects_custom_extensions(tmp_path):
    # A converter restricted to .png must ignore the .jpg images entirely.
    root = _build_image_folder(str(tmp_path / "images"))
    conv = ImageFolderConverter(extensions={".png"})
    with pytest.raises(ValueError, match="No image files"):
        conv.convert(root, str(tmp_path / "o.tar"), show_progress=False)


# ---------------------------------------------------------------------------
# PyTorchCompatibleLoader: iteration shape / dtype / labels
# ---------------------------------------------------------------------------


def _make_loader(dataset, **kw):
    kw.setdefault("pin_memory", False)  # avoid platform-specific pin_memory crash
    kw.setdefault("num_workers", 2)
    kw.setdefault(
        "label_extractor", FolderLabelExtractor(class_to_idx=dict(dataset["class_to_idx"]))
    )
    return PyTorchCompatibleLoader(dataset["tar"], **kw)


def test_iteration_yields_image_label_tuples(dataset):
    loader = _make_loader(dataset, batch_size=4, shuffle=False)
    batches = list(loader)
    assert len(batches) == 3  # 12 samples / batch 4
    for images, labels in batches:
        assert isinstance(images, torch.Tensor)
        assert isinstance(labels, torch.Tensor)
        # PyTorch layout: (N, C, H, W), default resize 224x224, float32 image.
        assert images.shape == (4, 3, 224, 224)
        assert images.dtype == torch.float32
        # Labels are 1-D int64 (torch.long), one per image.
        assert labels.shape == (4,)
        assert labels.dtype == torch.int64
        # Pixels are normalized into [0, 1].
        assert float(images.min()) >= 0.0
        assert float(images.max()) <= 1.0


def test_labels_follow_folder_class(dataset):
    # shuffle=False keeps sorted (ant, bee, cat) order: one class per batch.
    loader = _make_loader(dataset, batch_size=4, shuffle=False)
    seen = [labels.tolist() for _images, labels in loader]
    assert seen == [[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]]


def test_iteration_covers_every_sample_once(dataset):
    loader = _make_loader(dataset, batch_size=5, shuffle=False)
    all_labels = []
    total = 0
    for images, labels in loader:
        total += images.shape[0]
        all_labels.extend(labels.tolist())
    assert total == TOTAL
    # Each class contributes exactly PER_CLASS samples.
    assert sorted(all_labels) == sorted([0] * PER_CLASS + [1] * PER_CLASS + [2] * PER_CLASS)


def test_default_folder_extractor_is_deterministic(dataset):
    # With no explicit mapping, FolderLabelExtractor assigns indices in the
    # order classes are first seen; shuffle=False -> ant,bee,cat -> 0,1,2.
    loader = PyTorchCompatibleLoader(
        dataset["tar"], batch_size=4, shuffle=False, num_workers=2, pin_memory=False
    )
    seen = [labels.tolist() for _images, labels in loader]
    assert seen == [[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]]


def test_output_size_controls_spatial_dims(dataset):
    loader = _make_loader(dataset, batch_size=4, shuffle=False, output_size=(64, 48))
    images, _labels = next(iter(loader))
    assert images.shape == (4, 3, 64, 48)


def test_reiterable_across_epochs(dataset):
    loader = _make_loader(dataset, batch_size=4, shuffle=False)
    epoch1 = [labels.tolist() for _i, labels in loader]
    epoch2 = [labels.tolist() for _i, labels in loader]
    assert epoch1 == epoch2 == [[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]]


def test_shuffle_preserves_full_multiset(dataset):
    loader = _make_loader(dataset, batch_size=4, shuffle=True)
    loader.set_epoch(0)
    labels = []
    for _images, lb in loader:
        labels.extend(lb.tolist())
    # Shuffling must not drop or duplicate samples.
    assert len(labels) == TOTAL
    assert sorted(labels) == sorted([0] * PER_CLASS + [1] * PER_CLASS + [2] * PER_CLASS)


# ---------------------------------------------------------------------------
# PyTorchCompatibleLoader: len() / properties / set_epoch
# ---------------------------------------------------------------------------


def test_len_unknown_before_first_epoch_then_known(dataset):
    loader = _make_loader(dataset, batch_size=4, shuffle=False)
    # Before any iteration the batch count is unknown -> TypeError (PyTorch-like).
    with pytest.raises(TypeError, match="unknown until the first epoch"):
        len(loader)
    # After a full pass it reports the observed number of batches.
    consumed = list(loader)
    assert len(loader) == len(consumed) == 3


def test_loader_exposes_pytorch_properties(dataset):
    extractor = FolderLabelExtractor(class_to_idx=dict(dataset["class_to_idx"]))
    loader = PyTorchCompatibleLoader(
        dataset["tar"],
        batch_size=7,
        shuffle=False,
        num_workers=3,
        pin_memory=False,
        drop_last=True,
        label_extractor=extractor,
    )
    assert loader.batch_size == 7
    assert loader.num_workers == 3
    assert loader.pin_memory is False
    assert loader.drop_last is True
    assert loader.label_extractor is extractor
    # dataset property is a self-reference for PyTorch compatibility.
    assert loader.dataset is loader
    assert loader.sampler is None


def test_set_epoch_updates_internal_epoch(dataset):
    loader = _make_loader(dataset, batch_size=4, shuffle=True)
    loader.set_epoch(5)
    assert loader._epoch == 5


def test_context_manager_closes(dataset):
    with _make_loader(dataset, batch_size=4, shuffle=False) as loader:
        images, _labels = next(iter(loader))
        assert images.shape[0] == 4
    # close() ran on exit; calling it again must be safe (idempotent).
    loader.close()


def test_drop_last_drops_partial_batch(dataset):
    # 12 samples, batch_size=5 -> PyTorch would yield 2 full batches of 5 and
    # drop the remaining 2. Every yielded batch should be exactly batch_size.
    loader = _make_loader(dataset, batch_size=5, shuffle=False, drop_last=True)
    sizes = [images.shape[0] for images, _labels in loader]
    assert sizes == [5, 5], f"drop_last=True produced uneven batches: {sizes}"


def test_default_pin_memory_iterates(dataset):
    # Uses the pin_memory=True DEFAULT (note: no pin_memory kwarg here).
    loader = PyTorchCompatibleLoader(
        dataset["tar"],
        batch_size=4,
        shuffle=False,
        num_workers=1,
        label_extractor=FolderLabelExtractor(class_to_idx=dict(dataset["class_to_idx"])),
    )
    seen = [labels.tolist() for _images, labels in loader]
    assert seen == [[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]]


# ---------------------------------------------------------------------------
# TransformAdapter
# ---------------------------------------------------------------------------

torchvision = pytest.importorskip("torchvision")
import torchvision.transforms as T  # noqa: E402

import turboloader  # noqa: E402


def test_adapter_resize_equivalent_to_native(dataset):
    # Converting torchvision Resize must wire (H, W) into the native Resize so
    # that applying either produces byte-identical output.
    tv = T.Compose([T.Resize((64, 32))])
    adapted = TransformAdapter.from_torchvision(tv)
    assert type(adapted).__name__ == "Resize"
    assert adapted.name() == "Resize"

    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(48, 40, 3), dtype=np.uint8)
    native_out = turboloader.Resize(64, 32).apply(img)
    adapted_out = adapted.apply(img)
    assert adapted_out.shape == native_out.shape
    assert np.array_equal(adapted_out, native_out)


def test_adapter_multiple_transforms_compose(dataset):
    tv = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.RandomHorizontalFlip(0.5),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    out = TransformAdapter.from_torchvision(tv)
    assert type(out).__name__ == "ComposedTransforms"
    assert hasattr(out, "apply")


def test_adapter_single_transform_returns_bare_transform(dataset):
    out = TransformAdapter.from_torchvision(T.Compose([T.Grayscale()]))
    assert type(out).__name__ == "Grayscale"
    assert out.name() == "Grayscale"
    # Grayscale collapses to a single channel.
    img = np.full((16, 16, 3), 123, dtype=np.uint8)
    assert out.apply(img).shape == (16, 16, 1)


def test_adapter_empty_compose_returns_none(dataset):
    # ToPILImage has no turboloader equivalent -> nothing to compose -> None.
    out = TransformAdapter.from_torchvision(T.Compose([T.ToPILImage()]))
    assert out is None


def test_adapter_normalize_imagenet_detected(dataset):
    # ImageNet mean/std should map to the dedicated ImageNetNormalize op.
    tv = T.Compose([T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    out = TransformAdapter.from_torchvision(tv)
    assert type(out).__name__ == "ImageNetNormalize"


def test_adapter_imagenet_train_and_val_presets(dataset):
    train = TransformAdapter.imagenet_train()
    val = TransformAdapter.imagenet_val()
    assert type(train).__name__ == "ComposedTransforms"
    assert type(val).__name__ == "ComposedTransforms"
    assert hasattr(train, "apply") and hasattr(val, "apply")
