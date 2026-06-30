import time, numpy as np, torch, sys
from torchvision.datasets import ImageFolder
from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, IntField

class Fixed(ImageFolder):
    def __getitem__(self, i):
        img, label = super().__getitem__(i)
        return np.array(img.convert("RGB").resize((160, 160)), dtype=np.uint8), label

BETON = "/home/arnav/data/imagenette_fixed.beton"
if "--write" in sys.argv:
    ds = Fixed("/home/arnav/data/imagenette2-160/train")
    DatasetWriter(BETON, {"image": RGBImageField(max_resolution=160), "label": IntField()},
                  num_workers=8).from_indexed_dataset(ds)
    print("fixed beton written")
    sys.exit(0)

from ffcv.loader import Loader, OrderOption
from ffcv.fields.decoders import SimpleRGBImageDecoder
from ffcv.fields.basics import IntDecoder
from ffcv.transforms import ToTensor, ToDevice, ToTorchImage, NormalizeImage
SZ, BS, NW = 160, 64, 8
MEAN = np.array([0.485, 0.456, 0.406]) * 255
STD = np.array([0.229, 0.224, 0.225]) * 255
dev = torch.device("cuda:0")
img_pipe = [SimpleRGBImageDecoder(), ToTensor(), ToDevice(dev, non_blocking=True),
            ToTorchImage(), NormalizeImage(MEAN, STD, np.float32)]
lbl_pipe = [IntDecoder(), ToTensor(), ToDevice(dev)]
def timeit(epochs=3, warmup=1):
    rates = []
    for e in range(warmup + epochs):
        loader = Loader(BETON, batch_size=BS, num_workers=NW, order=OrderOption.SEQUENTIAL,
                        pipelines={"image": img_pipe, "label": lbl_pipe})
        n = 0; t0 = time.perf_counter()
        for x, y in loader:
            n += x.shape[0]; float(x.sum())
        dt = time.perf_counter() - t0
        if e >= warmup:
            rates.append(n / dt)
    return float(np.median(rates))
print(f"FFCV (fixed 160, gpu): {timeit():.0f} img/s")
