// Pure-C++ interface to the Metal video path (implemented in metal_video.mm).
// AVFoundation (AVAssetReader) drives VideoToolbox HARDWARE H.264/HEVC decode into
// NV12 CVPixelBuffers; a fused Metal kernel then does NV12 -> RGB (BT.601/709,
// video/full range) + bilinear resize + normalize -> CHW float32, one launch per
// frame, one command buffer per batch. No FFmpeg, no external deps — system
// frameworks only, so it ships in the ordinary macOS wheel.
//
// Same conventions as the resident loaders: process-global handle registry,
// double-buffered outputs (a returned pointer is valid until the NEXT call on the
// same handle), functions safe to call without the GIL.
#pragma once

#include <cstddef>

namespace turboloader {
namespace metal_video {

// True if Metal is up AND AVFoundation is available (always, on macOS arm64 builds).
bool available();

// Open a video for sequential hardware-accelerated decode.
//   frame_step: keep every Nth frame (1 = all; skipped frames are still decoded —
//               inter-frame codecs require it — just not converted/copied).
//   dst_h/dst_w: output size of the fused convert+resize; max_batch bounds later
//                next_batch() calls.
// Returns a handle >= 0, or -1 on failure (missing file, no video track, ...).
int open_video(const char* path, int frame_step, size_t max_batch, int dst_h, int dst_w);

// Best-effort count of frames next_batch() will deliver in total (ceil of
// track frame count / frame_step), or -1 if the container doesn't say.
long frame_count(int handle);

// Source video geometry / rate (after open). 0/0/0.0 on bad handle.
// src dims are re-latched from the first DECODED buffer (naturalSize can lie
// under rotation metadata), so query them after the first next_batch for truth.
int src_width(int handle);
int src_height(int handle);
double fps(int handle);

// Output geometry fixed at open — bindings validate view shapes against these.
int dst_width(int handle);
int dst_height(int handle);

// True if the stream stopped due to a decode/GPU error (vs clean end-of-stream).
bool has_failed(int handle);

// Decode + convert + resize + normalize the next `batch` kept frames.
// Writes (n, 3, dst_h, dst_w) float32 into the handle's double-buffered output.
// Returns n (0 = end of stream), sets *out to the buffer (valid until the next
// next_batch on this handle) and *first_index to the SOURCE frame index of out[0]
// (subsequent frames advance by frame_step).
size_t next_batch(int handle, size_t batch, const float mean[3], const float std_[3],
                  const float** out, long* first_index);

void close_video(int handle);

}  // namespace metal_video
}  // namespace turboloader
