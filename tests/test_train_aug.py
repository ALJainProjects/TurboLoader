"""Fused RandomResizedCrop + horizontal flip (train_aug=True) on the DirectBatch path.

The crop happens INSIDE the fused C++ pass (decode -> crop -> resize -> flip -> normalize),
so these tests recover the applied crop rect from a gradient image (R encodes x, G encodes
y): output R-range = crop x-extent, G-range = y-extent, R direction = flip. That lets us
verify torchvision-parity crop statistics without instrumenting the C++.
"""

import tarfile

import numpy as np
import pytest

import turboloader

Image = pytest.importorskip("PIL.Image")

N_IMGS = 64
SRC = 128  # gradient source size


@pytest.fixture(scope="module")
def gradient_tar(tmp_path_factory):
    root = tmp_path_factory.mktemp("trainaug")
    tar_path = str(root / "grad.tar")
    y, x = np.mgrid[0:SRC, 0:SRC]
    img = np.stack(
        [
            (x * 255 // (SRC - 1)),
            (y * 255 // (SRC - 1)),
            np.full_like(x, 128),
        ],
        axis=-1,
    ).astype(np.uint8)
    with tarfile.open(tar_path, "w") as tf:
        # DirectBatch decodes JPEG; quality=100 + no chroma subsampling keeps the
        # gradient accurate to ~1 level, well within the loose stat tolerances below.
        p = str(root / "g.jpg")
        Image.fromarray(img).save(p, quality=100, subsampling=0)
        for i in range(N_IMGS):
            tf.add(p, arcname=f"{i:03d}.jpg")
    return tar_path


def _crop_stats(batch):
    """Recover (scale_fraction, ratio, flipped) per image from gradient outputs."""
    out = []
    for im in batch:  # CHW in [0,1]
        r, g = im[0], im[1]
        flipped = bool(r[:, 0].mean() > r[:, -1].mean())
        rr = r[:, ::-1] if flipped else r
        x0, x1 = float(rr[:, 0].mean()), float(rr[:, -1].mean())
        y0, y1 = float(g[0, :].mean()), float(g[-1, :].mean())
        cw, ch = max(x1 - x0, 1e-3), max(y1 - y0, 1e-3)
        out.append((cw * ch, cw / ch, flipped))
    return out


def _loader(tar, seed=0, hflip=0.5):
    return turboloader.DataLoader(
        tar,
        batch_size=N_IMGS,
        output_format="pytorch",
        image_size=64,
        shuffle=False,
        seed=seed,
        train_aug=True,
        hflip_prob=hflip,
        prefetch_batches=0,
    )


def test_crop_distribution_matches_torchvision_params(gradient_tar):
    scales, ratios, flips = [], [], []
    loader = _loader(gradient_tar)
    for epoch in range(6):  # 384 samples
        loader.set_epoch(epoch)
        for im, _m in loader:
            for s, r, f in _crop_stats(np.asarray(im)):
                scales.append(s)
                ratios.append(r)
                flips.append(f)
    scales, ratios = np.array(scales), np.array(ratios)
    # torchvision RandomResizedCrop: scale ~ U(0.08, 1.0); ratio log-U(3/4, 4/3)
    assert scales.min() > 0.05 and scales.max() <= 1.02, (scales.min(), scales.max())
    assert 0.45 < scales.mean() < 0.65, f"scale mean {scales.mean():.3f} != ~0.54 (U(0.08,1))"
    assert ratios.min() > 0.70 and ratios.max() < 1.40, (ratios.min(), ratios.max())
    # log-ratio symmetric around 0
    assert abs(np.log(ratios).mean()) < 0.06, np.log(ratios).mean()
    flip_rate = np.mean(flips)
    assert 0.40 < flip_rate < 0.60, f"flip rate {flip_rate:.2f} != ~0.5"


def test_deterministic_per_epoch_different_across_epochs(gradient_tar):
    a = _loader(gradient_tar, seed=7)
    a.set_epoch(3)
    b1 = np.asarray(next(iter(a))[0]).copy()
    a.set_epoch(3)
    b2 = np.asarray(next(iter(a))[0]).copy()
    a.set_epoch(4)
    b3 = np.asarray(next(iter(a))[0]).copy()
    assert np.array_equal(b1, b2), "same seed+epoch must reproduce identical crops"
    assert not np.array_equal(b1, b3), "different epochs must produce different crops"


def test_hflip_prob_zero_and_one(gradient_tar):
    none = _crop_stats(np.asarray(next(iter(_loader(gradient_tar, hflip=0.0)))[0]))
    all_ = _crop_stats(np.asarray(next(iter(_loader(gradient_tar, hflip=1.0)))[0]))
    assert not any(f for _s, _r, f in none)
    assert all(f for _s, _r, f in all_)


def test_train_aug_rejects_inexpressible_pipelines(gradient_tar):
    with pytest.raises(ValueError):
        turboloader.DataLoader(
            gradient_tar,
            batch_size=8,
            output_format="pytorch",
            image_size=64,
            train_aug=True,
            transform=turboloader.ColorJitter(0.4, 0.4, 0.4),
        )
