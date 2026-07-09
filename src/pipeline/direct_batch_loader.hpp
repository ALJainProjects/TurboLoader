#pragma once

// Direct-batch image loader (FFCV / tf.data-style).
//
// Unlike the producer/consumer pipeline (TarWorker -> SPSC queue -> assembly), this
// path runs ONE data-parallel pass per batch: each thread reads a sample's JPEG bytes
// by index, DCT-scaled-decodes, resizes to the exact target, and fused-normalizes
// DIRECTLY into its slot of the output batch buffer. No queue, no per-sample heap
// allocation, no single-threaded collection step. This removes the per-sample overhead
// that otherwise dominates when decode is cheap (large images + scaled decode).

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <numeric>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "../core/compat.hpp"        // turboloader::span (portable C++17/20, no raw <span>)
#include "../core/parallel_for.hpp"
#include "../decode/jpeg_decoder.hpp"
#include "../readers/tar_reader.hpp"
#include "../transforms/simd_utils.hpp"

namespace turboloader {

class DirectBatchCore {
public:
    DirectBatchCore(const std::string& data_path,
                    std::shared_ptr<std::vector<uint8_t>> remote_data,
                    size_t batch_size, int target_h, int target_w,
                    bool chw, bool normalize_01,
                    const std::vector<float>& mean, const std::vector<float>& std_vec,
                    bool shuffle, int seed,
                    bool distributed, int world_rank, int world_size,
                    bool drop_last = false, bool antialias = false,
                    bool train_aug = false, float hflip_prob = 0.5f)
        : batch_size_(batch_size), target_h_(target_h), target_w_(target_w),
          chw_(chw), normalize_(normalize_01), shuffle_(shuffle), seed_(seed),
          distributed_(distributed), world_rank_(world_rank), world_size_(world_size),
          drop_last_(drop_last), antialias_(antialias), train_aug_(train_aug),
          hflip_prob_(hflip_prob), epoch_(0), cursor_(0) {
        if (remote_data) {
            reader_ = std::make_unique<TarReader>(remote_data, 0, 1);
        } else {
            reader_ = std::make_unique<TarReader>(data_path, 0, 1);
        }
        total_ = reader_->num_samples();
        if (mean.size() == 3 && std_vec.size() == 3) {
            mean_ = mean;
            inv_std_ = {1.0f / std_vec[0], 1.0f / std_vec[1], 1.0f / std_vec[2]};
            has_mean_std_ = true;
        }
        build_order();
    }

    // Number of samples actually yielded per epoch (honors drop_last).
    // Samples that failed to decode and were served as zero-filled images. Silent black
    // images poison training invisibly, so failures are counted, warned to stderr (first
    // few), and surfaced to Python in every batch's metadata.
    size_t decode_failures() const { return decode_failures_.load(std::memory_order_relaxed); }

    size_t num_samples() const {
        size_t n = order_.size();
        if (drop_last_) return (n / batch_size_) * batch_size_;
        return n;
    }
    size_t batch_size() const { return batch_size_; }
    int target_h() const { return target_h_; }
    int target_w() const { return target_w_; }
    bool chw() const { return chw_; }

    // Start a fresh epoch: (re)build the (optionally shuffled / sharded) order and
    // reset the batch cursor.
    // start_batch > 0 resumes mid-epoch: the deterministic order (seed/epoch/shard) is
    // rebuilt identically and the cursor skips the first start_batch batches — exact,
    // decode-free resumption (powers DataLoader.state_dict()/load_state_dict()).
    void begin_epoch(int epoch, size_t start_batch = 0) {
        epoch_ = epoch;
        build_order();
        cursor_.store(start_batch * batch_size_, std::memory_order_relaxed);
    }

    // Atomically claim the next batch's sample indices. Returns the batch size
    // (0 means the epoch is exhausted). Thread-safe.
    size_t acquire_batch(std::vector<size_t>& indices) {
        size_t start = cursor_.fetch_add(batch_size_, std::memory_order_relaxed);
        const size_t n = order_.size();
        if (start >= n) return 0;
        // drop_last: skip the ragged final batch entirely.
        if (drop_last_ && start + batch_size_ > n) return 0;
        size_t bs = std::min(batch_size_, n - start);
        indices.assign(order_.begin() + start, order_.begin() + start + bs);
        return bs;
    }

    // Decode+resize+normalize every sample in `indices` directly into `out`
    // (preallocated: bs * C * H * W floats, CHW or HWC per chw_). Runs data-parallel
    // on the global thread pool. Caller must NOT hold the GIL.
    void fill_batch(const std::vector<size_t>& indices, float* out) {
        const int H = target_h_, W = target_w_, C = 3;
        const size_t num_pixels = static_cast<size_t>(H) * W;
        const size_t per = num_pixels * C;
        parallel_for(indices.size(), [&](size_t i) {
            process_one(indices[i], out + i * per, H, W, C, num_pixels);
        });
    }

private:
    void build_order() {
        std::vector<size_t> all(total_);
        std::iota(all.begin(), all.end(), 0);
        if (shuffle_) {
            std::mt19937_64 rng(static_cast<uint64_t>(seed_) + static_cast<uint64_t>(epoch_));
            std::shuffle(all.begin(), all.end(), rng);
        }
        if (distributed_ && world_size_ > 1) {
            // Same shuffle seed on every rank => identical `all` => disjoint shards.
            // EQUAL shards on every rank (drop up to world_size-1 trailing samples)
            // so all ranks yield the same batch count — otherwise DDP collectives
            // deadlock when one rank runs out early. Matches DistributedSampler(drop_last).
            size_t per_rank = total_ / static_cast<size_t>(world_size_);
            size_t s = static_cast<size_t>(world_rank_) * per_rank;
            size_t e = s + per_rank;
            order_.assign(all.begin() + s, all.begin() + e);
        } else {
            order_ = std::move(all);
        }
    }

    // torchvision-parity RandomResizedCrop rect on a (w, h) image: 10 attempts of
    // area*U(0.08,1) with log-uniform aspect in [3/4, 4/3], center-crop fallback.
    // Seeded by (seed, epoch, idx): deterministic per sample per epoch, different across
    // epochs — the exact reproducibility contract of the shuffle path.
    struct CropRect { int x, y, w, h; bool flip; };
    CropRect pick_crop(int w, int h, size_t idx) const {
        std::mt19937_64 rng((static_cast<uint64_t>(seed_) << 32) ^
                            (static_cast<uint64_t>(epoch_) << 20) ^ static_cast<uint64_t>(idx));
        std::uniform_real_distribution<float> uni(0.0f, 1.0f);
        const float area = static_cast<float>(w) * h;
        CropRect r{0, 0, w, h, false};
        for (int attempt = 0; attempt < 10; ++attempt) {
            float target_area = area * (0.08f + uni(rng) * (1.0f - 0.08f));
            float log_lo = std::log(3.0f / 4.0f), log_hi = std::log(4.0f / 3.0f);
            float ratio = std::exp(log_lo + uni(rng) * (log_hi - log_lo));
            int cw = static_cast<int>(std::lround(std::sqrt(target_area * ratio)));
            int ch = static_cast<int>(std::lround(std::sqrt(target_area / ratio)));
            if (cw > 0 && ch > 0 && cw <= w && ch <= h) {
                r.w = cw;
                r.h = ch;
                r.x = static_cast<int>(uni(rng) * (w - cw + 1));
                r.y = static_cast<int>(uni(rng) * (h - ch + 1));
                if (r.x > w - cw) r.x = w - cw;
                if (r.y > h - ch) r.y = h - ch;
                r.flip = uni(rng) < hflip_prob_;
                return r;
            }
        }
        // Fallback (torchvision): central crop at the clamped aspect ratio.
        float in_ratio = static_cast<float>(w) / h;
        if (in_ratio < 3.0f / 4.0f) {
            r.w = w;
            r.h = std::min(h, static_cast<int>(std::lround(w / (3.0f / 4.0f))));
        } else if (in_ratio > 4.0f / 3.0f) {
            r.h = h;
            r.w = std::min(w, static_cast<int>(std::lround(h * (4.0f / 3.0f))));
        }
        r.x = (w - r.w) / 2;
        r.y = (h - r.h) / 2;
        r.flip = uni(rng) < hflip_prob_;
        return r;
    }

    static void hflip_hwc_inplace(uint8_t* img, int w, int h, int c) {
        for (int y = 0; y < h; ++y) {
            uint8_t* row = img + static_cast<size_t>(y) * w * c;
            for (int x0 = 0, x1 = w - 1; x0 < x1; ++x0, --x1) {
                for (int k = 0; k < c; ++k) std::swap(row[x0 * c + k], row[x1 * c + k]);
            }
        }
    }

    void process_one(size_t idx, float* out, int H, int W, int C, size_t num_pixels) {
        // Per-thread decoder (libjpeg cinfo is not shareable) and scratch buffers,
        // persisted across batches by the persistent thread pool.
        thread_local JPEGDecoder decoder;
        thread_local std::vector<uint8_t> decoded;
        thread_local std::vector<uint8_t> resized;

        int aw = 0, ah = 0, ch = 0;
        try {
            turboloader::span<const uint8_t> jpeg = reader_->get_sample(idx);  // const, thread-safe
            decoder.decode_scaled(jpeg, decoded, W, H, aw, ah, ch);
        } catch (...) {
            record_decode_failure(idx, "decode failed (corrupt or unsupported file)");
            std::memset(out, 0, num_pixels * C * sizeof(float));
            return;
        }
        if (ch != C) {
            // Unexpected channel count (e.g. CMYK/gray) — zero-fill rather than corrupt.
            record_decode_failure(idx, "unexpected channel count (e.g. CMYK/grayscale)");
            std::memset(out, 0, num_pixels * C * sizeof(float));
            return;
        }

        if (train_aug_) {
            // Fused RandomResizedCrop + horizontal flip. The crop rect is chosen on the
            // DCT-scaled decoded image (aw x ah) — proportionally identical to choosing it
            // on the original, and every crop pixel keeps >= target resolution because the
            // scaled decode already covers the FULL image at target size. Crop rows are
            // packed contiguous (memcpy, small vs decode) so the SIMD resize applies.
            thread_local std::vector<uint8_t> cropped;
            CropRect r = pick_crop(aw, ah, idx);
            cropped.resize(static_cast<size_t>(r.w) * r.h * C);
            for (int y = 0; y < r.h; ++y) {
                std::memcpy(cropped.data() + static_cast<size_t>(y) * r.w * C,
                            decoded.data() + ((static_cast<size_t>(r.y) + y) * aw + r.x) * C,
                            static_cast<size_t>(r.w) * C);
            }
            resized.resize(static_cast<size_t>(W) * H * C);
            if (r.w == W && r.h == H) {
                std::memcpy(resized.data(), cropped.data(), resized.size());
            } else if (antialias_) {
                transforms::simd::resize_triangle_antialias(cropped.data(), resized.data(),
                                                            r.w, r.h, W, H, C);
            } else {
                transforms::simd::resize_bilinear_simd(cropped.data(), resized.data(),
                                                       r.w, r.h, W, H, C);
            }
            if (r.flip) hflip_hwc_inplace(resized.data(), W, H, C);
            const uint8_t* src_aug = resized.data();
            if (chw_) {
                transforms::simd::deinterleave_hwc_to_chw_f32(
                    src_aug, out, out + num_pixels, out + 2 * num_pixels,
                    num_pixels, normalize_,
                    has_mean_std_ ? mean_.data() : nullptr,
                    has_mean_std_ ? inv_std_.data() : nullptr);
            } else {
                const float scale = normalize_ ? (1.0f / 255.0f) : 1.0f;
                if (has_mean_std_) {
                    for (size_t p = 0; p < num_pixels; ++p) {
                        for (int c = 0; c < C; ++c) {
                            out[p * C + c] =
                                (src_aug[p * C + c] * scale - mean_[c]) * inv_std_[c];
                        }
                    }
                } else {
                    for (size_t e = 0; e < num_pixels * C; ++e) out[e] = src_aug[e] * scale;
                }
            }
            return;
        }

        const uint8_t* src;
        if (aw != W || ah != H) {
            resized.resize(static_cast<size_t>(W) * H * C);
            if (antialias_) {
                transforms::simd::resize_triangle_antialias(decoded.data(), resized.data(),
                                                            aw, ah, W, H, C);
            } else {
                transforms::simd::resize_bilinear_simd(decoded.data(), resized.data(),
                                                       aw, ah, W, H, C);
            }
            src = resized.data();
        } else {
            src = decoded.data();
        }

        if (chw_) {
            transforms::simd::deinterleave_hwc_to_chw_f32(
                src, out, out + num_pixels, out + 2 * num_pixels,
                num_pixels, normalize_,
                has_mean_std_ ? mean_.data() : nullptr,
                has_mean_std_ ? inv_std_.data() : nullptr);
        } else {
            // HWC float32 output
            const float scale = normalize_ ? (1.0f / 255.0f) : 1.0f;
            if (has_mean_std_) {
                for (size_t p = 0; p < num_pixels; ++p) {
                    for (int c = 0; c < C; ++c) {
                        out[p * C + c] = (src[p * C + c] * scale - mean_[c]) * inv_std_[c];
                    }
                }
            } else {
                for (size_t e = 0; e < num_pixels * C; ++e) out[e] = src[e] * scale;
            }
        }
    }

    std::unique_ptr<TarReader> reader_;
    size_t total_ = 0;
    size_t batch_size_;
    int target_h_, target_w_;
    bool chw_, normalize_;
    bool has_mean_std_ = false;
    std::vector<float> mean_, inv_std_;
    bool shuffle_;
    int seed_;
    bool distributed_;
    int world_rank_, world_size_;
    bool drop_last_;
    bool antialias_;
    bool train_aug_ = false;   // fused RandomResizedCrop(scale=[0.08,1], ratio=[3/4,4/3]) + hflip
    float hflip_prob_ = 0.5f;
    int epoch_;
    std::vector<size_t> order_;
    void record_decode_failure(size_t idx, const char* why) {
        size_t n = decode_failures_.fetch_add(1, std::memory_order_relaxed) + 1;
        if (n <= 5) {  // rate-limit: warn for the first few, count the rest
            std::fprintf(stderr,
                         "[turboloader] WARNING: sample %zu %s; serving a zero-filled image. "
                         "(failure %zu%s)\n",
                         idx, why, n, n == 5 ? "; further warnings suppressed" : "");
        }
    }

    std::atomic<size_t> decode_failures_{0};
    std::atomic<size_t> cursor_;
};

}  // namespace turboloader
