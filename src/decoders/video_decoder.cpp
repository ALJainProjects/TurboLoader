#include "turboloader/decoders/video_decoder.hpp"
#include <stdexcept>
#include <cstring>

#ifdef HAVE_FFMPEG
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>
}
#endif

namespace turboloader {

#ifdef HAVE_FFMPEG

struct VideoDecoder::Impl {
    Config config_;

    Impl(const Config& config) : config_(config) {}

    std::vector<DecodedVideoFrame> decode_video(std::span<const uint8_t> data) {
        std::vector<DecodedVideoFrame> frames;

        // Create custom I/O context for memory buffer
        AVIOContext* avio_ctx = create_avio_context(data);
        if (!avio_ctx) {
            throw std::runtime_error("Failed to create AVIO context");
        }

        // Open format context
        AVFormatContext* fmt_ctx = avformat_alloc_context();
        fmt_ctx->pb = avio_ctx;

        if (avformat_open_input(&fmt_ctx, nullptr, nullptr, nullptr) < 0) {
            av_free(avio_ctx->buffer);
            avio_context_free(&avio_ctx);
            throw std::runtime_error("Failed to open video");
        }

        if (avformat_find_stream_info(fmt_ctx, nullptr) < 0) {
            avformat_close_input(&fmt_ctx);
            av_free(avio_ctx->buffer);
            avio_context_free(&avio_ctx);
            throw std::runtime_error("Failed to find stream info");
        }

        // Find video stream
        int video_stream_idx = -1;
        AVCodecParameters* codec_params = nullptr;

        for (unsigned int i = 0; i < fmt_ctx->nb_streams; i++) {
            if (fmt_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
                video_stream_idx = i;
                codec_params = fmt_ctx->streams[i]->codecpar;
                break;
            }
        }

        if (video_stream_idx == -1) {
            avformat_close_input(&fmt_ctx);
            av_free(avio_ctx->buffer);
            avio_context_free(&avio_ctx);
            throw std::runtime_error("No video stream found");
        }

        // Find decoder
        const AVCodec* codec = avcodec_find_decoder(codec_params->codec_id);
        if (!codec) {
            avformat_close_input(&fmt_ctx);
            av_free(avio_ctx->buffer);
            avio_context_free(&avio_ctx);
            throw std::runtime_error("Codec not found");
        }

        AVCodecContext* codec_ctx = avcodec_alloc_context3(codec);
        if (avcodec_parameters_to_context(codec_ctx, codec_params) < 0) {
            avcodec_free_context(&codec_ctx);
            avformat_close_input(&fmt_ctx);
            av_free(avio_ctx->buffer);
            avio_context_free(&avio_ctx);
            throw std::runtime_error("Failed to copy codec params");
        }

        if (avcodec_open2(codec_ctx, codec, nullptr) < 0) {
            avcodec_free_context(&codec_ctx);
            avformat_close_input(&fmt_ctx);
            av_free(avio_ctx->buffer);
            avio_context_free(&avio_ctx);
            throw std::runtime_error("Failed to open codec");
        }

        // Setup frame conversion to RGB
        SwsContext* sws_ctx = sws_getContext(
            codec_ctx->width, codec_ctx->height, codec_ctx->pix_fmt,
            config_.target_width > 0 ? config_.target_width : codec_ctx->width,
            config_.target_height > 0 ? config_.target_height : codec_ctx->height,
            AV_PIX_FMT_RGB24,
            SWS_BILINEAR, nullptr, nullptr, nullptr
        );

        int out_width = config_.target_width > 0 ? config_.target_width : codec_ctx->width;
        int out_height = config_.target_height > 0 ? config_.target_height : codec_ctx->height;

        // Decode frames
        AVPacket* packet = av_packet_alloc();
        AVFrame* frame = av_frame_alloc();
        AVFrame* rgb_frame = av_frame_alloc();

        rgb_frame->format = AV_PIX_FMT_RGB24;
        rgb_frame->width = out_width;
        rgb_frame->height = out_height;
        av_frame_get_buffer(rgb_frame, 0);

        int frame_count = 0;
        double time_base = av_q2d(fmt_ctx->streams[video_stream_idx]->time_base);

        while (av_read_frame(fmt_ctx, packet) >= 0) {
            if (packet->stream_index != video_stream_idx) {
                av_packet_unref(packet);
                continue;
            }

            if (avcodec_send_packet(codec_ctx, packet) < 0) {
                av_packet_unref(packet);
                continue;
            }

            while (avcodec_receive_frame(codec_ctx, frame) == 0) {
                double timestamp = frame->pts * time_base;

                // Check if we should skip this frame based on target FPS
                if (config_.target_fps > 0) {
                    double frame_interval = 1.0 / config_.target_fps;
                    if (frame_count > 0 && timestamp < frame_count * frame_interval) {
                        continue;
                    }
                }

                // Convert to RGB
                sws_scale(sws_ctx,
                         frame->data, frame->linesize, 0, codec_ctx->height,
                         rgb_frame->data, rgb_frame->linesize);

                // Extract RGB data
                DecodedVideoFrame decoded_frame;
                decoded_frame.width = out_width;
                decoded_frame.height = out_height;
                decoded_frame.channels = 3;
                decoded_frame.timestamp = timestamp;
                decoded_frame.frame_number = frame_count;

                size_t data_size = out_width * out_height * 3;
                decoded_frame.data.resize(data_size);

                // Copy RGB data (handle potential padding in linesize)
                for (int y = 0; y < out_height; y++) {
                    memcpy(decoded_frame.data.data() + y * out_width * 3,
                           rgb_frame->data[0] + y * rgb_frame->linesize[0],
                           out_width * 3);
                }

                frames.push_back(std::move(decoded_frame));
                frame_count++;

                if (config_.max_frames > 0 && frame_count >= config_.max_frames) {
                    break;
                }
            }

            av_packet_unref(packet);

            if (config_.max_frames > 0 && frame_count >= config_.max_frames) {
                break;
            }
        }

        // Cleanup
        av_frame_free(&rgb_frame);
        av_frame_free(&frame);
        av_packet_free(&packet);
        sws_freeContext(sws_ctx);
        avcodec_free_context(&codec_ctx);
        avformat_close_input(&fmt_ctx);
        av_free(avio_ctx->buffer);
        avio_context_free(&avio_ctx);

        return frames;
    }

    VideoInfo get_video_info(std::span<const uint8_t> data) {
        AVIOContext* avio_ctx = create_avio_context(data);
        AVFormatContext* fmt_ctx = avformat_alloc_context();
        fmt_ctx->pb = avio_ctx;

        if (avformat_open_input(&fmt_ctx, nullptr, nullptr, nullptr) < 0) {
            av_free(avio_ctx->buffer);
            avio_context_free(&avio_ctx);
            throw std::runtime_error("Failed to open video for info");
        }

        if (avformat_find_stream_info(fmt_ctx, nullptr) < 0) {
            avformat_close_input(&fmt_ctx);
            av_free(avio_ctx->buffer);
            avio_context_free(&avio_ctx);
            throw std::runtime_error("Failed to find stream info");
        }

        VideoInfo info{};

        for (unsigned int i = 0; i < fmt_ctx->nb_streams; i++) {
            if (fmt_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
                AVStream* stream = fmt_ctx->streams[i];
                info.width = stream->codecpar->width;
                info.height = stream->codecpar->height;
                info.duration = fmt_ctx->duration / (double)AV_TIME_BASE;
                info.fps = av_q2d(stream->avg_frame_rate);
                info.total_frames = stream->nb_frames;

                const AVCodec* codec = avcodec_find_decoder(stream->codecpar->codec_id);
                if (codec) {
                    info.codec = codec->name;
                }
                break;
            }
        }

        avformat_close_input(&fmt_ctx);
        av_free(avio_ctx->buffer);
        avio_context_free(&avio_ctx);

        return info;
    }

private:
    struct IOContext {
        const uint8_t* data;
        size_t size;
        size_t pos;
    };

    static int read_packet(void* opaque, uint8_t* buf, int buf_size) {
        IOContext* io_ctx = static_cast<IOContext*>(opaque);

        if (io_ctx->pos >= io_ctx->size) {
            return AVERROR_EOF;
        }

        size_t to_read = std::min(static_cast<size_t>(buf_size), io_ctx->size - io_ctx->pos);
        memcpy(buf, io_ctx->data + io_ctx->pos, to_read);
        io_ctx->pos += to_read;

        return to_read;
    }

    static int64_t seek_packet(void* opaque, int64_t offset, int whence) {
        IOContext* io_ctx = static_cast<IOContext*>(opaque);

        if (whence == AVSEEK_SIZE) {
            return io_ctx->size;
        }

        if (whence == SEEK_SET) {
            io_ctx->pos = offset;
        } else if (whence == SEEK_CUR) {
            io_ctx->pos += offset;
        } else if (whence == SEEK_END) {
            io_ctx->pos = io_ctx->size + offset;
        }

        io_ctx->pos = std::min(io_ctx->pos, io_ctx->size);

        return io_ctx->pos;
    }

    AVIOContext* create_avio_context(std::span<const uint8_t> data) {
        IOContext* io_ctx = new IOContext{data.data(), data.size(), 0};

        size_t buffer_size = 4096;
        uint8_t* buffer = static_cast<uint8_t*>(av_malloc(buffer_size));

        AVIOContext* avio_ctx = avio_alloc_context(
            buffer, buffer_size, 0, io_ctx,
            &read_packet, nullptr, &seek_packet
        );

        return avio_ctx;
    }
};

#else  // !HAVE_FFMPEG

struct VideoDecoder::Impl {
    Config config_;

    Impl(const Config& config) : config_(config) {}

    std::vector<DecodedVideoFrame> decode_video(std::span<const uint8_t> data) {
        throw std::runtime_error(
            "Video decoder not available - install FFmpeg and rebuild with -DHAVE_FFMPEG");
    }

    VideoInfo get_video_info(std::span<const uint8_t> data) {
        throw std::runtime_error(
            "Video decoder not available - install FFmpeg and rebuild with -DHAVE_FFMPEG");
    }
};

#endif  // HAVE_FFMPEG

VideoDecoder::VideoDecoder()
    : pimpl_(std::make_unique<Impl>(Config{})) {
}

VideoDecoder::VideoDecoder(const Config& config)
    : pimpl_(std::make_unique<Impl>(config)) {
}

VideoDecoder::~VideoDecoder() = default;
VideoDecoder::VideoDecoder(VideoDecoder&&) noexcept = default;
VideoDecoder& VideoDecoder::operator=(VideoDecoder&&) noexcept = default;

std::vector<DecodedVideoFrame> VideoDecoder::decode(std::span<const uint8_t> data) {
    return pimpl_->decode_video(data);
}

DecodedVideoFrame VideoDecoder::decode_frame(std::span<const uint8_t> data, double timestamp) {
    Config config = pimpl_->config_;
    config.start_time = timestamp;
    config.max_frames = 1;

    Impl impl(config);
    auto frames = impl.decode_video(data);

    if (frames.empty()) {
        throw std::runtime_error("No frame at timestamp " + std::to_string(timestamp));
    }

    return frames[0];
}

VideoDecoder::VideoInfo VideoDecoder::get_info(std::span<const uint8_t> data) {
    return pimpl_->get_video_info(data);
}

bool VideoDecoder::is_video(std::span<const uint8_t> data) {
    if (data.size() < 12) return false;

    // Check for common video file signatures
    // MP4: ftyp
    if (data.size() >= 8 && data[4] == 'f' && data[5] == 't' &&
        data[6] == 'y' && data[7] == 'p') {
        return true;
    }

    // AVI: RIFF...AVI
    if (data[0] == 'R' && data[1] == 'I' && data[2] == 'F' && data[3] == 'F' &&
        data[8] == 'A' && data[9] == 'V' && data[10] == 'I') {
        return true;
    }

    // WebM/MKV: EBML
    if (data[0] == 0x1A && data[1] == 0x45 && data[2] == 0xDF && data[3] == 0xA3) {
        return true;
    }

    return false;
}

}  // namespace turboloader
