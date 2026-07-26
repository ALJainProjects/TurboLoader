# TBL v2 Binary Format


TurboLoader includes a custom binary format optimized for ML workloads:

### Features
- LZ4 compression for reduced storage
- Memory-mapped access for fast loading
- O(1) random access via indexed structure
- Data integrity validation with CRC checksums
- Cached image dimensions for filtered loading

### Convert TAR to TBL

```python
import tarfile
import turboloader

writer = turboloader.TblWriterV2("/data/imagenet.tbl", enable_compression=True)

# The TAR archive is read with Python's stdlib (TurboLoader does not expose a
# standalone Python TarReader; the DataLoader reads TAR directly for training).
with tarfile.open("/data/imagenet.tar") as tar:
    for member in tar.getmembers():
        if not member.name.lower().endswith((".jpg", ".jpeg")):
            continue
        data = tar.extractfile(member).read()
        writer.add_sample(data=data, format=turboloader.SampleFormat.JPEG)

writer.finalize()
```

> For bulk conversion there is also a C++ CLI tool, `tools/tar_to_tbl_v2.cpp`.
