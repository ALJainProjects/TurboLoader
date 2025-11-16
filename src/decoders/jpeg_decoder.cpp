#include "turboloader/decoders/jpeg_decoder.hpp"
#include <jpeglib.h>
#include <csetjmp>
#include <cstring>
#include <stdexcept>

namespace turboloader {

// Custom error handler for libjpeg
struct JpegErrorManager {
    jpeg_error_mgr pub;
    jmp_buf setjmp_buffer;
    char message[JMSG_LENGTH_MAX];
};

static void jpeg_error_exit(j_common_ptr cinfo) {
    JpegErrorManager* err = reinterpret_cast<JpegErrorManager*>(cinfo->err);
    (*cinfo->err->format_message)(cinfo, err->message);
    longjmp(err->setjmp_buffer, 1);
}

struct JpegDecoder::Impl {
    jpeg_decompress_struct cinfo;
    JpegErrorManager jerr;

    Impl() {
        // Set up error handling
        cinfo.err = jpeg_std_error(&jerr.pub);
        jerr.pub.error_exit = jpeg_error_exit;

        jpeg_create_decompress(&cinfo);
    }

    ~Impl() {
        jpeg_destroy_decompress(&cinfo);
    }
};

JpegDecoder::JpegDecoder()
    : pimpl_(std::make_unique<Impl>()) {
}

JpegDecoder::~JpegDecoder() = default;

JpegDecoder::JpegDecoder(JpegDecoder&&) noexcept = default;
JpegDecoder& JpegDecoder::operator=(JpegDecoder&&) noexcept = default;

DecodedImage JpegDecoder::decode(std::span<const uint8_t> jpeg_data) {
    if (jpeg_data.empty()) {
        throw std::invalid_argument("Empty JPEG data");
    }

    // Set up error handling
    if (setjmp(pimpl_->jerr.setjmp_buffer)) {
        // JPEG error occurred - clean up before throwing
        jpeg_abort_decompress(&pimpl_->cinfo);
        throw std::runtime_error(std::string("JPEG decode error: ") + pimpl_->jerr.message);
    }

    // Set source to memory buffer
    jpeg_mem_src(&pimpl_->cinfo,
                 const_cast<unsigned char*>(jpeg_data.data()),
                 jpeg_data.size());

    // Read JPEG header
    if (jpeg_read_header(&pimpl_->cinfo, TRUE) != JPEG_HEADER_OK) {
        jpeg_abort_decompress(&pimpl_->cinfo);
        throw std::runtime_error("Invalid JPEG header");
    }

    // Force RGB output
    pimpl_->cinfo.out_color_space = JCS_RGB;

    // Start decompression
    jpeg_start_decompress(&pimpl_->cinfo);

    int width = pimpl_->cinfo.output_width;
    int height = pimpl_->cinfo.output_height;
    int channels = pimpl_->cinfo.output_components;

    if (channels != 3) {
        jpeg_abort_decompress(&pimpl_->cinfo);
        throw std::runtime_error("Expected 3 channels (RGB), got " + std::to_string(channels));
    }

    // Allocate output buffer
    DecodedImage result;
    result.width = width;
    result.height = height;
    result.channels = channels;
    result.data.resize(width * height * channels);

    // Read scanlines
    int row_stride = width * channels;
    JSAMPROW row_pointer[1];

    while (pimpl_->cinfo.output_scanline < pimpl_->cinfo.output_height) {
        row_pointer[0] = &result.data[pimpl_->cinfo.output_scanline * row_stride];
        jpeg_read_scanlines(&pimpl_->cinfo, row_pointer, 1);
    }

    // Finish decompression - this properly cleans up internal state
    jpeg_finish_decompress(&pimpl_->cinfo);

    return result;
}

bool JpegDecoder::can_decode(std::span<const uint8_t> data) const {
    return is_jpeg(data);
}

bool JpegDecoder::is_jpeg(std::span<const uint8_t> data) {
    // Check for JPEG magic bytes: FF D8 FF
    return data.size() >= 3 &&
           data[0] == 0xFF &&
           data[1] == 0xD8 &&
           data[2] == 0xFF;
}

}  // namespace turboloader
