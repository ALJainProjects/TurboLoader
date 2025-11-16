#pragma once

#include <cstdint>
#include <span>
#include <vector>
#include <memory>
#include <string>

namespace turboloader {

/**
 * Decoded video frame
 */
struct DecodedVideoFrame {
    std::vector<uint8_t> data;  // RGB pixels (HWC)
    int width{0};
    int height{0};
    int channels{3};
    double timestamp{0.0};  // Frame timestamp in seconds
    int frame_number{0};
};

/**
 * Video decoder using FFmpeg
 *
 * Features:
 * - H.264, H.265, VP9, AV1 support
 * - Frame extraction
 * - Temporal sampling
 * - Hardware acceleration (when available)
 */
class VideoDecoder {
public:
    struct Config {
        // Target FPS for frame extraction (0 = all frames)
        double target_fps{0.0};

        // Maximum frames to extract (0 = all)
        size_t max_frames{0};

        // Start time in seconds
        double start_time{0.0};

        // Duration in seconds (0 = to end)
        double duration{0.0};

        // Use hardware acceleration if available
        bool use_hw_accel{true};

        // Resize frames to this size (0 = original)
        int target_width{0};
        int target_height{0};
    };

    VideoDecoder();
    explicit VideoDecoder(const Config& config);
    ~VideoDecoder();

    // Non-copyable but movable
    VideoDecoder(const VideoDecoder&) = delete;
    VideoDecoder& operator=(const VideoDecoder&) = delete;
    VideoDecoder(VideoDecoder&&) noexcept;
    VideoDecoder& operator=(VideoDecoder&&) noexcept;

    /**
     * Decode video from memory buffer
     * @return Vector of decoded frames
     */
    std::vector<DecodedVideoFrame> decode(std::span<const uint8_t> data);

    /**
     * Decode single frame at timestamp
     */
    DecodedVideoFrame decode_frame(std::span<const uint8_t> data, double timestamp);

    /**
     * Get video information without decoding
     */
    struct VideoInfo {
        int width;
        int height;
        double duration;  // seconds
        double fps;
        int total_frames;
        std::string codec;
    };

    VideoInfo get_info(std::span<const uint8_t> data);

    /**
     * Check if data is a supported video format
     */
    static bool is_video(std::span<const uint8_t> data);

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};

}  // namespace turboloader
