# TensorFlow integration

TurboLoader is framework-agnostic: `output_format="tensorflow"` yields `(N, H, W, 3)` float32 HWC batches (numpy), and `output_format="numpy"` the same layout. Feed them to `tf.data.Dataset.from_generator`:

```python
import numpy as np, tensorflow as tf, turboloader as tl

labels = np.load("labels.npy")                         # aligned to the TAR's member order
loader = tl.DataLoader("train.tar", batch_size=64, output_format="tensorflow", image_size=224,
                       transform=tl.ImageNetNormalize(), shuffle=True, seed=0, train_aug=True)

def gen():
    for epoch in range(epochs):
        loader.set_epoch(epoch)
        for x, meta in loader:                          # x: (64, 224, 224, 3) float32 numpy
            yield x, labels[np.asarray(meta["indices"])]

ds = tf.data.Dataset.from_generator(
    gen,
    output_signature=(tf.TensorSpec((None, 224, 224, 3), tf.float32), tf.TensorSpec((None,), tf.int64)),
).prefetch(tf.data.AUTOTUNE)
model.fit(ds, epochs=epochs, steps_per_epoch=len(loader))
```

Notes:
- Samples carry no label key; align your label array by `meta['indices']`.
- Measured against `tf.data` itself on Imagenette-160 (Apple M4 Max, real consumption): TurboLoader on-the-fly ~55k img/s vs `tf.data` (AUTOTUNE) ~27k — methodology and caveats in [docs/benchmarks](../benchmarks/index.md).
- Keep the loader on the fast path (`output_format` set, `image_size` fixed, transforms fused) — per-sample Python work is what makes input pipelines slow.
- For many-epoch training with a fixed recipe, `preprocess_to_tbl` + `DataLoader("x.tbl", output_format="tensorflow")` skips decode entirely (HWC output via `numpy`/`tensorflow` formats is a transpose of the CHW serve — use `TblRawImageLoader` directly for CHW).
- A known TensorFlow shutdown quirk ("mutex lock failed" at exit) is on TF's side; exit with `sys.exit(0)` after `gc.collect()`.

Canonical docs: the [project wiki](https://github.com/ALJainProjects/TurboLoader/wiki).
