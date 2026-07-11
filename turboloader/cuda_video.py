"""CUDA video loader (NVIDIA) — decoded video to GPU-resident training batches.

Two decode backends feed the same fused CUDA kernel (YUV 4:2:0 -> RGB video-range
BT.601/709 + bilinear resize + normalize, the numpy-validated Metal kernel's math):

- ``decode="cpu"`` (default): threaded FFmpeg decode via PyAV on the host, one
  H2D upload per batch, conversion + resize + normalize on the GPU. Frees the
  CPU of all color/resize work and lands batches on-device where training wants
  them. This is the right default because CPU decode is FASTER than NVDEC on
  some stacks — measured on an RTX 3090 under WSL2: PyAV 1,453 f/s vs NVDEC
  130 f/s (video engine virtualization overhead).

- ``decode="nvdec"``: PyNvVideoCodec hardware decode straight to device NV12,
  ZERO host->device traffic. The right choice on native Linux / datacenter GPUs
  where NVDEC runs at spec; measure both on your machine (the benchmark script
  does) and pick.

Yields GPU-resident ``(B, 3, H, W)`` float32 batches (``__cuda_array_interface__``,
adopt zero-copy with ``torch.as_tensor(x, device="cuda")``). A yielded batch is
valid until the NEXT batch (double-buffered) — copy or consume before advancing.
"""

import numpy as np

__all__ = ["CudaVideoLoader"]


class CudaVideoLoader:
    def __init__(
        self,
        path,
        image_size=224,
        batch_size=32,
        *,
        decode="cpu",
        frame_step=1,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        return_indices=False,
        gpu_id=0,
    ):
        import turboloader as t

        if not getattr(t, "cuda_available", lambda: False)() or not hasattr(
            t, "cuda_video_yuv420_batch"
        ):
            raise RuntimeError("CudaVideoLoader needs a CUDA build with cuda_video_yuv420_batch.")
        if decode not in ("cpu", "nvdec"):
            raise ValueError("decode must be 'cpu' or 'nvdec'")
        self._t = t
        self.path = str(path)
        h, w = (image_size, image_size) if isinstance(image_size, int) else image_size
        self._h, self._w = int(h), int(w)
        self.batch_size = int(batch_size)
        self.decode = decode
        self.frame_step = int(frame_step)
        self.mean = list(mean)
        self.std = list(std)
        self.return_indices = bool(return_indices)
        self.gpu_id = int(gpu_id)

    def _wrap(self, ptr, n):
        from turboloader.cuda_loader import _CudaArray

        return _CudaArray(ptr, (n, 3, self._h, self._w))

    # ------------------------- CPU (PyAV) backend --------------------------
    def _iter_cpu(self):
        import av
        import torch

        with av.open(self.path) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            W, H = stream.codec_context.width, stream.codec_context.height
            if W % 2 or H % 2:
                raise ValueError("yuv420p video must have even dimensions")
            bt709 = self._bt709(stream, W, H)
            ysz, csz = W * H, (W // 2) * (H // 2)
            fbytes = ysz + 2 * csz

            host = np.empty((self.batch_size, fbytes), dtype=np.uint8)
            batch_rows = 0
            first = -1
            idx = 0
            for frame in container.decode(stream):
                keep = idx % self.frame_step == 0
                if keep:
                    if frame.format.name != "yuv420p":
                        frame = frame.reformat(format="yuv420p")
                    host[batch_rows] = frame.to_ndarray().reshape(-1)  # clean packed I420
                    if batch_rows == 0:
                        first = idx
                    batch_rows += 1
                idx += 1
                if batch_rows == self.batch_size:
                    yield self._flush_cpu(host, batch_rows, W, H, bt709, first, torch)
                    batch_rows = 0
            if batch_rows:
                yield self._flush_cpu(host, batch_rows, W, H, bt709, first, torch)

    def _flush_cpu(self, host, n, W, H, bt709, first, torch):
        dev = torch.from_numpy(host[:n]).cuda()  # one H2D per batch
        base = int(dev.data_ptr())
        ysz, csz = W * H, (W // 2) * (H // 2)
        fbytes = ysz + 2 * csz
        y = [base + i * fbytes for i in range(n)]
        cb = [p + ysz for p in y]
        cr = [p + ysz + csz for p in y]
        ptr = self._t.cuda_video_yuv420_batch(
            y,
            cb,
            cr,
            y_stride=W,
            c_stride=W // 2,
            c_px_stride=1,
            src_w=W,
            src_h=H,
            dst_h=self._h,
            dst_w=self._w,
            bt709=bt709,
            mean=self.mean,
            std=self.std,
        )
        del dev  # kernel is synchronous: upload no longer needed
        batch = self._wrap(ptr, n)
        if self.return_indices:
            return batch, np.arange(first, first + n * self.frame_step, self.frame_step)
        return batch

    @staticmethod
    def _bt709(stream, W, H):
        cs = getattr(getattr(stream, "codec_context", None), "colorspace", None)
        name = str(cs).lower() if cs is not None else ""
        if "709" in name:
            return True
        if "601" in name or "170" in name or "470" in name:
            return False
        return H >= 720 or W >= 1280  # untagged: same heuristic as the Metal path

    # ------------------------- NVDEC backend -------------------------------
    def _iter_nvdec(self):
        import PyNvVideoCodec as nvc

        demux = nvc.CreateDemuxer(self.path)
        dec = nvc.CreateDecoder(
            gpuid=self.gpu_id,
            codec=demux.GetNvCodecId(),
            cudacontext=0,
            cudastream=0,
            usedevicememory=True,
        )
        frames, first, idx = [], -1, 0
        W = H = None
        bt709 = False
        for packet in demux:
            for frame in dec.Decode(packet):
                keep = idx % self.frame_step == 0
                if keep:
                    if W is None:
                        cai = frame.cuda().__cuda_array_interface__
                        H = cai["shape"][0] * 2 // 3  # NV12: (H*3/2, W)
                        W = cai["shape"][1]
                        bt709 = H >= 720 or W >= 1280
                    if first < 0:
                        first = idx
                    frames.append(frame)  # keep alive until the kernel consumed them
                idx += 1
                if len(frames) == self.batch_size:
                    yield self._flush_nvdec(frames, W, H, bt709, first)
                    frames, first = [], -1
        if frames:
            yield self._flush_nvdec(frames, W, H, bt709, first)

    def _flush_nvdec(self, frames, W, H, bt709, first):
        y, cb, cr = [], [], []
        y_stride = None
        for f in frames:
            cai = f.cuda().__cuda_array_interface__
            base = int(cai["data"][0])
            stride = cai["strides"][0] if cai.get("strides") else W
            y_stride = int(stride)
            y.append(base)
            cbcr = base + H * y_stride  # NV12: interleaved CbCr plane after Y
            cb.append(cbcr)
            cr.append(cbcr + 1)
        n = len(frames)
        ptr = self._t.cuda_video_yuv420_batch(
            y,
            cb,
            cr,
            y_stride=y_stride,
            c_stride=y_stride,
            c_px_stride=2,
            src_w=W,
            src_h=H,
            dst_h=self._h,
            dst_w=self._w,
            bt709=bt709,
            mean=self.mean,
            std=self.std,
        )
        batch = self._wrap(ptr, n)
        if self.return_indices:
            return batch, np.arange(first, first + n * self.frame_step, self.frame_step)
        return batch

    def __iter__(self):
        return self._iter_cpu() if self.decode == "cpu" else self._iter_nvdec()
