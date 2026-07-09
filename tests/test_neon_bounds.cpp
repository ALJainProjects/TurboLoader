// Regression test: NEON HSV batch kernels must not touch memory past the pixel buffer.
//
// Bug history (<= 2.31.0): rgb_to_hsv_batch_neon / hsv_to_rgb_batch_neon advanced their
// vector loops 4 pixels per iteration but used vld3_u8/vst3_u8, which ALWAYS access
// 8 pixels (24 bytes) — so any num_pixels >= 4 overread/overwrote up to 12 bytes past the
// end of the buffer on the final vector iteration (heap corruption via ColorJitter /
// adjust_saturation on ARM, including the published macOS arm64 wheels).
//
// NOTE: AddressSanitizer canNOT catch this class of bug — vst3_u8 lowers to the
// llvm.aarch64.neon.st3 intrinsic, which ASan does not instrument. Hence explicit
// guard-byte checking here. On non-ARM builds the NEON path doesn't exist and this test
// trivially passes (the scalar path is bounds-exact by construction).

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "../src/transforms/simd_utils.hpp"

namespace ts = turboloader::transforms::simd;

namespace {

constexpr uint8_t kSentinel = 0xAB;
constexpr size_t kGuard = 64;

// Returns the number of corrupted guard bytes after running fn on an exact-size buffer.
template <typename Fn>
int guarded_run(size_t num_pixels, Fn&& fn) {
    const size_t data = num_pixels * 3;
    uint8_t* buf = new uint8_t[data + kGuard];
    std::memset(buf, kSentinel, data + kGuard);
    for (size_t i = 0; i < data; i++) buf[i] = static_cast<uint8_t>(i * 7);
    fn(buf, num_pixels);
    int corrupted = 0;
    for (size_t i = data; i < data + kGuard; i++)
        if (buf[i] != kSentinel) corrupted++;
    delete[] buf;
    return corrupted;
}

}  // namespace

int main() {
#if defined(TURBOLOADER_SIMD_NEON)
    // Sizes that end exactly on a vector-loop boundary are the dangerous ones (the last
    // iteration is where the 8-px access runs past a 4-px remainder); also test odd sizes.
    const size_t sizes[] = {4, 8, 12, 20, 33, 64, 100, 257};
    for (size_t n : sizes) {
        // hsv_to_rgb: the overwrite direction (writes RGB).
        int bad_w = guarded_run(n, [](uint8_t* rgb, size_t np) {
            float* h = new float[np];
            float* s = new float[np];
            float* v = new float[np];
            for (size_t i = 0; i < np; i++) {
                h[i] = static_cast<float>((i * 17) % 360);
                s[i] = 0.5f;
                v[i] = 0.5f;
            }
            ts::hsv_to_rgb_batch_neon(h, s, v, rgb, np);
            delete[] h;
            delete[] s;
            delete[] v;
        });
        if (bad_w != 0) {
            std::fprintf(stderr, "FAIL: hsv_to_rgb_batch_neon overwrote %d guard bytes at n=%zu\n",
                         bad_w, n);
            return 1;
        }
        // rgb_to_hsv: the overread direction. Overreads don't corrupt the guard, but the
        // staged-load fix also guarantees the outputs only depend on in-bounds bytes:
        // flipping guard bytes must not change results.
        const size_t data = n * 3;
        uint8_t* buf = new uint8_t[data + kGuard];
        float* h1 = new float[n];
        float* s1 = new float[n];
        float* v1 = new float[n];
        float* h2 = new float[n];
        float* s2 = new float[n];
        float* v2 = new float[n];
        for (size_t i = 0; i < data; i++) buf[i] = static_cast<uint8_t>(i * 13);
        std::memset(buf + data, 0x00, kGuard);
        ts::rgb_to_hsv_batch_neon(buf, h1, s1, v1, n);
        std::memset(buf + data, 0xFF, kGuard);  // change ONLY out-of-bounds bytes
        ts::rgb_to_hsv_batch_neon(buf, h2, s2, v2, n);
        if (std::memcmp(h1, h2, n * sizeof(float)) != 0 ||
            std::memcmp(s1, s2, n * sizeof(float)) != 0 ||
            std::memcmp(v1, v2, n * sizeof(float)) != 0) {
            std::fprintf(stderr, "FAIL: rgb_to_hsv_batch_neon output depends on out-of-bounds "
                                 "bytes at n=%zu (overread)\n", n);
            return 1;
        }
        delete[] buf;
        delete[] h1; delete[] s1; delete[] v1;
        delete[] h2; delete[] s2; delete[] v2;
    }
    std::printf("PASS: NEON HSV kernels are bounds-exact for all tested sizes\n");
#else
    std::printf("PASS (trivial): NEON path not compiled on this architecture\n");
#endif
    return 0;
}
