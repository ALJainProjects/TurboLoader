#include "turboloader/decoders/gpu_jpeg_decoder.hpp"
#include <stdexcept>
#include <cstring>

#ifdef TURBOLOADER_WITH_CUDA
#include <cuda_runtime.h>
#include <nvjpeg.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
        } \
    } while(0)

#define NVJPEG_CHECK(call) \
    do { \
        nvjpegStatus_t status = call; \
        if (status != NVJPEG_STATUS_SUCCESS) { \
            throw std::runtime_error(std::string("nvJPEG error: ") + std::to_string(status)); \
        } \
    } while(0)
#endif

namespace turboloader {

#ifdef TURBOLOADER_WITH_CUDA
struct GpuJpegDecoder::Impl {
    nvjpegHandle_t nvjpeg_handle;
    nvjpegJpegState_t jpeg_state;
    nvjpegBufferPinned_t pinned_buffer;
    nvjpegBufferDevice_t device_buffer;
    nvjpegJpegStream_t jpeg_stream;
    nvjpegDecodeParams_t decode_params;

    cudaStream_t cuda_stream;
    int device_id;
    Config config;

    // Batch decoding state
    std::vector<nvjpegImage_t> decode_outputs;
    std::vector<const unsigned char*> batched_inputs;
    std::vector<size_t> batched_input_sizes;

    Impl(const Config& cfg) : config(cfg), device_id(cfg.device_id) {
        // Set CUDA device
        CUDA_CHECK(cudaSetDevice(device_id));

        // Create nvJPEG handle
        NVJPEG_CHECK(nvjpegCreateSimple(&nvjpeg_handle));

        // Create JPEG state for decoding
        NVJPEG_CHECK(nvjpegJpegStateCreate(nvjpeg_handle, &jpeg_state));

        // Create decode parameters
        NVJPEG_CHECK(nvjpegDecodeParamsCreate(nvjpeg_handle, &decode_params));

        // Set RGB output format
        NVJPEG_CHECK(nvjpegDecodeParamsSetOutputFormat(decode_params, NVJPEG_OUTPUT_RGBI));

        // Create pinned and device buffers
        NVJPEG_CHECK(nvjpegBufferPinnedCreate(nvjpeg_handle, nullptr, &pinned_buffer));
        NVJPEG_CHECK(nvjpegBufferDeviceCreate(nvjpeg_handle, nullptr, &device_buffer));

        // Create JPEG stream
        NVJPEG_CHECK(nvjpegJpegStreamCreate(nvjpeg_handle, &jpeg_stream));

        // Create CUDA stream if enabled
        if (config.use_cuda_stream) {
            CUDA_CHECK(cudaStreamCreate(&cuda_stream));
        } else {
            cuda_stream = nullptr;
        }

        // Pre-allocate batch vectors
        decode_outputs.reserve(config.max_batch_size);
        batched_inputs.reserve(config.max_batch_size);
        batched_input_sizes.reserve(config.max_batch_size);
    }

    ~Impl() {
        if (cuda_stream) {
            cudaStreamDestroy(cuda_stream);
        }
        nvjpegJpegStreamDestroy(jpeg_stream);
        nvjpegBufferDeviceDestroy(device_buffer);
        nvjpegBufferPinnedDestroy(pinned_buffer);
        nvjpegDecodeParamsDestroy(decode_params);
        nvjpegJpegStateDestroy(jpeg_state);
        nvjpegDestroy(nvjpeg_handle);
    }

    DecodedImage decode_single(std::span<const uint8_t> jpeg_data) {
        CUDA_CHECK(cudaSetDevice(device_id));

        // Parse JPEG header to get dimensions
        int nComponents, subsampling, widths[NVJPEG_MAX_COMPONENT], heights[NVJPEG_MAX_COMPONENT];
        nvjpegChromaSubsampling_t chroma_subsampling;

        NVJPEG_CHECK(nvjpegGetImageInfo(
            nvjpeg_handle,
            jpeg_data.data(),
            jpeg_data.size(),
            &nComponents,
            &chroma_subsampling,
            widths,
            heights
        ));

        int width = widths[0];
        int height = heights[0];
        int channels = 3;  // RGB

        // Allocate GPU memory for output
        nvjpegImage_t output_image;
        size_t pitch = width * channels;
        size_t buffer_size = pitch * height;

        CUDA_CHECK(cudaMalloc(&output_image.channel[0], buffer_size));
        output_image.pitch[0] = pitch;

        // Decode JPEG to GPU
        NVJPEG_CHECK(nvjpegDecode(
            nvjpeg_handle,
            jpeg_state,
            jpeg_data.data(),
            jpeg_data.size(),
            NVJPEG_OUTPUT_RGBI,
            &output_image,
            cuda_stream
        ));

        // Synchronize if using streams
        if (cuda_stream) {
            CUDA_CHECK(cudaStreamSynchronize(cuda_stream));
        }

        // Create DecodedImage with GPU memory
        DecodedImage result;
        result.width = width;
        result.height = height;
        result.channels = channels;
        result.data.resize(buffer_size);

        // Copy from GPU to CPU (user can skip this for GPU training)
        CUDA_CHECK(cudaMemcpy(
            result.data.data(),
            output_image.channel[0],
            buffer_size,
            cudaMemcpyDeviceToHost
        ));

        // Store GPU pointer in metadata (for zero-copy GPU training)
        result.device_ptr = output_image.channel[0];

        return result;
    }
};

#else
// CPU-only fallback implementation
struct GpuJpegDecoder::Impl {
    Impl(const Config&) {
        throw std::runtime_error("GPU decoder not available - TurboLoader not compiled with CUDA support");
    }

    DecodedImage decode_single(std::span<const uint8_t>) {
        throw std::runtime_error("GPU decoder not available");
    }
};
#endif

// GpuJpegDecoder implementation
GpuJpegDecoder::GpuJpegDecoder(const Config& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

GpuJpegDecoder::~GpuJpegDecoder() = default;

GpuJpegDecoder::GpuJpegDecoder(GpuJpegDecoder&&) noexcept = default;
GpuJpegDecoder& GpuJpegDecoder::operator=(GpuJpegDecoder&&) noexcept = default;

DecodedImage GpuJpegDecoder::decode(std::span<const uint8_t> jpeg_data) {
#ifdef TURBOLOADER_WITH_CUDA
    return pimpl_->decode_single(jpeg_data);
#else
    throw std::runtime_error("GPU decoder not available - TurboLoader not compiled with CUDA support");
#endif
}

std::vector<DecodedImage> GpuJpegDecoder::decode_batch(
    const std::vector<std::span<const uint8_t>>& jpeg_batches) {
#ifdef TURBOLOADER_WITH_CUDA
    std::vector<DecodedImage> results;
    results.reserve(jpeg_batches.size());

    // For now, decode sequentially (TODO: implement true batch decode)
    for (const auto& jpeg_data : jpeg_batches) {
        results.push_back(pimpl_->decode_single(jpeg_data));
    }

    return results;
#else
    throw std::runtime_error("GPU decoder not available - TurboLoader not compiled with CUDA support");
#endif
}

bool GpuJpegDecoder::can_decode(std::span<const uint8_t> data) const {
    // Check JPEG magic bytes
    if (data.size() < 3) return false;
    return data[0] == 0xFF && data[1] == 0xD8 && data[2] == 0xFF;
}

void* GpuJpegDecoder::get_device_ptr(const DecodedImage& image) const {
    return image.device_ptr;
}

void GpuJpegDecoder::synchronize() {
#ifdef TURBOLOADER_WITH_CUDA
    if (pimpl_->cuda_stream) {
        CUDA_CHECK(cudaStreamSynchronize(pimpl_->cuda_stream));
    }
#endif
}

bool GpuJpegDecoder::is_available() {
#ifdef TURBOLOADER_WITH_CUDA
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return err == cudaSuccess && device_count > 0;
#else
    return false;
#endif
}

int GpuJpegDecoder::get_device_count() {
#ifdef TURBOLOADER_WITH_CUDA
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess) return 0;
    return device_count;
#else
    return 0;
#endif
}

// GpuImageBuffer implementation
GpuImageBuffer::GpuImageBuffer(int device_id, size_t width, size_t height, size_t channels)
    : device_id_(device_id), size_(width * height * channels) {
#ifdef TURBOLOADER_WITH_CUDA
    CUDA_CHECK(cudaSetDevice(device_id_));
    CUDA_CHECK(cudaMalloc(&device_ptr_, size_));
#else
    throw std::runtime_error("GPU buffers not available - TurboLoader not compiled with CUDA support");
#endif
}

GpuImageBuffer::~GpuImageBuffer() {
#ifdef TURBOLOADER_WITH_CUDA
    if (device_ptr_) {
        cudaFree(device_ptr_);
    }
#endif
}

GpuImageBuffer::GpuImageBuffer(GpuImageBuffer&& other) noexcept
    : device_ptr_(other.device_ptr_), size_(other.size_), device_id_(other.device_id_) {
    other.device_ptr_ = nullptr;
    other.size_ = 0;
}

GpuImageBuffer& GpuImageBuffer::operator=(GpuImageBuffer&& other) noexcept {
#ifdef TURBOLOADER_WITH_CUDA
    if (this != &other) {
        if (device_ptr_) {
            cudaFree(device_ptr_);
        }
        device_ptr_ = other.device_ptr_;
        size_ = other.size_;
        device_id_ = other.device_id_;
        other.device_ptr_ = nullptr;
        other.size_ = 0;
    }
#endif
    return *this;
}

std::vector<uint8_t> GpuImageBuffer::to_cpu() const {
#ifdef TURBOLOADER_WITH_CUDA
    std::vector<uint8_t> cpu_data(size_);
    CUDA_CHECK(cudaMemcpy(cpu_data.data(), device_ptr_, size_, cudaMemcpyDeviceToHost));
    return cpu_data;
#else
    throw std::runtime_error("GPU buffers not available");
#endif
}

void GpuImageBuffer::from_cpu(std::span<const uint8_t> data) {
#ifdef TURBOLOADER_WITH_CUDA
    if (data.size() != size_) {
        throw std::runtime_error("Data size mismatch");
    }
    CUDA_CHECK(cudaMemcpy(device_ptr_, data.data(), size_, cudaMemcpyHostToDevice));
#else
    throw std::runtime_error("GPU buffers not available");
#endif
}

}  // namespace turboloader
