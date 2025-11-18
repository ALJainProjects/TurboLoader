/**
 * @file image_dimensions.hpp
 * @brief Fast image dimension detection without full decode
 *
 * Reads image headers to extract width/height without decoding pixel data.
 * Supports: JPEG, PNG, WebP, BMP, TIFF
 */

#pragma once

#include "../formats/tbl_v2_format.hpp"
#include <cstdint>
#include <cstring>
#include <utility>

namespace turboloader {
namespace utils {

/**
 * @brief Read big-endian uint16
 */
inline uint16_t read_be16(const uint8_t* data) {
    return (static_cast<uint16_t>(data[0]) << 8) | data[1];
}

/**
 * @brief Read big-endian uint32
 */
inline uint32_t read_be32(const uint8_t* data) {
    return (static_cast<uint32_t>(data[0]) << 24) |
           (static_cast<uint32_t>(data[1]) << 16) |
           (static_cast<uint32_t>(data[2]) << 8) |
           data[3];
}

/**
 * @brief Read little-endian uint16
 */
inline uint16_t read_le16(const uint8_t* data) {
    return data[0] | (static_cast<uint16_t>(data[1]) << 8);
}

/**
 * @brief Read little-endian uint32
 */
inline uint32_t read_le32(const uint8_t* data) {
    return data[0] |
           (static_cast<uint32_t>(data[1]) << 8) |
           (static_cast<uint32_t>(data[2]) << 16) |
           (static_cast<uint32_t>(data[3]) << 24);
}

/**
 * @brief Detect JPEG dimensions
 *
 * Scans for SOF (Start of Frame) marker
 */
inline std::pair<uint16_t, uint16_t> detect_jpeg_dimensions(
    const uint8_t* data, size_t size)
{
    if (size < 3 || data[0] != 0xFF || data[1] != 0xD8) {
        return {0, 0};
    }

    size_t pos = 2;
    while (pos + 9 < size) {
        // Find next marker
        if (data[pos] != 0xFF) {
            pos++;
            continue;
        }

        uint8_t marker = data[pos + 1];
        pos += 2;

        // Skip padding bytes
        while (pos < size && marker == 0xFF) {
            marker = data[pos++];
        }

        // SOF markers (Start of Frame)
        // 0xC0 - Baseline DCT
        // 0xC1 - Extended sequential DCT
        // 0xC2 - Progressive DCT
        // 0xC3 - Lossless (sequential)
        if ((marker >= 0xC0 && marker <= 0xC3) ||
            (marker >= 0xC5 && marker <= 0xC7) ||
            (marker >= 0xC9 && marker <= 0xCB) ||
            (marker >= 0xCD && marker <= 0xCF)) {

            if (pos + 7 > size) break;

            // Skip length (2 bytes) and precision (1 byte)
            pos += 3;

            // Read height and width (big-endian)
            uint16_t height = read_be16(data + pos);
            uint16_t width = read_be16(data + pos + 2);

            return {width, height};
        }

        // Read segment length and skip
        if (pos + 2 > size) break;
        uint16_t length = read_be16(data + pos);
        pos += length;
    }

    return {0, 0};
}

/**
 * @brief Detect PNG dimensions
 *
 * Reads IHDR chunk
 */
inline std::pair<uint16_t, uint16_t> detect_png_dimensions(
    const uint8_t* data, size_t size)
{
    // PNG signature: 89 50 4E 47 0D 0A 1A 0A
    if (size < 24 || data[0] != 0x89 || data[1] != 0x50 ||
        data[2] != 0x4E || data[3] != 0x47) {
        return {0, 0};
    }

    // IHDR chunk is always first after signature (8 bytes)
    // IHDR format: [length:4][type:4][data:13][crc:4]
    // Data: width:4, height:4, bit_depth:1, color_type:1, ...

    size_t pos = 8;  // Skip signature
    if (pos + 8 > size) return {0, 0};

    // Verify IHDR chunk type
    if (data[pos + 4] != 'I' || data[pos + 5] != 'H' ||
        data[pos + 6] != 'D' || data[pos + 7] != 'R') {
        return {0, 0};
    }

    // Read width and height (big-endian)
    uint32_t width = read_be32(data + pos + 8);
    uint32_t height = read_be32(data + pos + 12);

    // Clamp to uint16_t range
    if (width > 65535) width = 65535;
    if (height > 65535) height = 65535;

    return {static_cast<uint16_t>(width), static_cast<uint16_t>(height)};
}

/**
 * @brief Detect WebP dimensions
 */
inline std::pair<uint16_t, uint16_t> detect_webp_dimensions(
    const uint8_t* data, size_t size)
{
    // WebP format: RIFF [size] WEBP [format]
    if (size < 30 || data[0] != 'R' || data[1] != 'I' ||
        data[2] != 'F' || data[3] != 'F') {
        return {0, 0};
    }

    if (data[8] != 'W' || data[9] != 'E' ||
        data[10] != 'B' || data[11] != 'P') {
        return {0, 0};
    }

    // VP8 (lossy)
    if (data[12] == 'V' && data[13] == 'P' && data[14] == '8' && data[15] == ' ') {
        if (size < 30) return {0, 0};

        // Skip frame tag (3 bytes starting at offset 23)
        size_t pos = 26;
        if (pos + 4 > size) return {0, 0};

        // Read dimensions from bitstream
        uint16_t width = (read_le16(data + pos) & 0x3FFF);
        uint16_t height = (read_le16(data + pos + 2) & 0x3FFF);

        return {width, height};
    }
    // VP8L (lossless)
    else if (data[12] == 'V' && data[13] == 'P' && data[14] == '8' && data[15] == 'L') {
        if (size < 25) return {0, 0};

        // Read dimensions from bitstream (packed format)
        uint32_t bits = read_le32(data + 21);
        uint16_t width = (bits & 0x3FFF) + 1;
        uint16_t height = ((bits >> 14) & 0x3FFF) + 1;

        return {width, height};
    }
    // VP8X (extended)
    else if (data[12] == 'V' && data[13] == 'P' && data[14] == '8' && data[15] == 'X') {
        if (size < 30) return {0, 0};

        // Dimensions are 24-bit values
        uint32_t width = (data[24] | (data[25] << 8) | (data[26] << 16)) + 1;
        uint32_t height = (data[27] | (data[28] << 8) | (data[29] << 16)) + 1;

        if (width > 65535) width = 65535;
        if (height > 65535) height = 65535;

        return {static_cast<uint16_t>(width), static_cast<uint16_t>(height)};
    }

    return {0, 0};
}

/**
 * @brief Detect BMP dimensions
 */
inline std::pair<uint16_t, uint16_t> detect_bmp_dimensions(
    const uint8_t* data, size_t size)
{
    // BMP signature: BM
    if (size < 26 || data[0] != 'B' || data[1] != 'M') {
        return {0, 0};
    }

    // DIB header starts at offset 14
    // Width is at offset 18 (4 bytes, little-endian)
    // Height is at offset 22 (4 bytes, little-endian)

    uint32_t width = read_le32(data + 18);
    uint32_t height = read_le32(data + 22);

    // Handle negative height (top-down BMP)
    if (height & 0x80000000) {
        height = -static_cast<int32_t>(height);
    }

    // Clamp to uint16_t range
    if (width > 65535) width = 65535;
    if (height > 65535) height = 65535;

    return {static_cast<uint16_t>(width), static_cast<uint16_t>(height)};
}

/**
 * @brief Detect TIFF dimensions
 */
inline std::pair<uint16_t, uint16_t> detect_tiff_dimensions(
    const uint8_t* data, size_t size)
{
    if (size < 8) return {0, 0};

    // Determine byte order
    bool little_endian;
    if (data[0] == 'I' && data[1] == 'I') {
        little_endian = true;
    } else if (data[0] == 'M' && data[1] == 'M') {
        little_endian = false;
    } else {
        return {0, 0};
    }

    // Read IFD offset
    uint32_t ifd_offset = little_endian ? read_le32(data + 4) : read_be32(data + 4);

    if (ifd_offset + 2 > size) return {0, 0};

    // Read number of directory entries
    uint16_t num_entries = little_endian ?
        read_le16(data + ifd_offset) :
        read_be16(data + ifd_offset);

    // Search for width (tag 256) and height (tag 257)
    uint32_t width = 0, height = 0;
    size_t entry_offset = ifd_offset + 2;

    for (uint16_t i = 0; i < num_entries && entry_offset + 12 <= size; ++i) {
        const uint8_t* entry = data + entry_offset;

        uint16_t tag = little_endian ? read_le16(entry) : read_be16(entry);
        uint16_t type = little_endian ? read_le16(entry + 2) : read_be16(entry + 2);

        // Read value (offset 8-11 in entry)
        uint32_t value = little_endian ? read_le32(entry + 8) : read_be32(entry + 8);

        if (tag == 256) {  // ImageWidth
            width = value;
        } else if (tag == 257) {  // ImageLength (height)
            height = value;
        }

        entry_offset += 12;

        if (width > 0 && height > 0) break;
    }

    // Clamp to uint16_t range
    if (width > 65535) width = 65535;
    if (height > 65535) height = 65535;

    return {static_cast<uint16_t>(width), static_cast<uint16_t>(height)};
}

/**
 * @brief Detect image dimensions from raw data
 *
 * @param data Image data pointer
 * @param size Image data size
 * @param format Image format (optional, will auto-detect if UNKNOWN)
 * @return Pair of (width, height). Returns (0, 0) on failure.
 */
inline std::pair<uint16_t, uint16_t> detect_image_dimensions(
    const uint8_t* data, size_t size,
    formats::SampleFormat format = formats::SampleFormat::UNKNOWN)
{
    if (!data || size < 16) {
        return {0, 0};
    }

    // Auto-detect format if unknown
    if (format == formats::SampleFormat::UNKNOWN) {
        // JPEG magic: FF D8 FF
        if (size >= 3 && data[0] == 0xFF && data[1] == 0xD8 && data[2] == 0xFF) {
            format = formats::SampleFormat::JPEG;
        }
        // PNG magic: 89 50 4E 47 0D 0A 1A 0A
        else if (size >= 8 && data[0] == 0x89 && data[1] == 0x50 &&
                 data[2] == 0x4E && data[3] == 0x47) {
            format = formats::SampleFormat::PNG;
        }
        // WebP magic: RIFF .... WEBP
        else if (size >= 12 && data[0] == 'R' && data[1] == 'I' &&
                 data[2] == 'F' && data[3] == 'F' &&
                 data[8] == 'W' && data[9] == 'E' &&
                 data[10] == 'B' && data[11] == 'P') {
            format = formats::SampleFormat::WEBP;
        }
        // BMP magic: BM
        else if (size >= 2 && data[0] == 'B' && data[1] == 'M') {
            format = formats::SampleFormat::BMP;
        }
        // TIFF magic: II or MM
        else if (size >= 4 && ((data[0] == 'I' && data[1] == 'I') ||
                               (data[0] == 'M' && data[1] == 'M'))) {
            format = formats::SampleFormat::TIFF;
        }
    }

    // Parse based on format
    switch (format) {
        case formats::SampleFormat::JPEG:
            return detect_jpeg_dimensions(data, size);
        case formats::SampleFormat::PNG:
            return detect_png_dimensions(data, size);
        case formats::SampleFormat::WEBP:
            return detect_webp_dimensions(data, size);
        case formats::SampleFormat::BMP:
            return detect_bmp_dimensions(data, size);
        case formats::SampleFormat::TIFF:
            return detect_tiff_dimensions(data, size);
        default:
            return {0, 0};
    }
}

} // namespace utils
} // namespace turboloader
