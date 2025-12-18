/**
 * @file test_audio_decoder.cpp
 * @brief Tests for audio decoding support (v2.19.0)
 *
 * Tests:
 * - WAV decoding (8/16/24/32-bit PCM, IEEE float)
 * - Format detection (WAV, FLAC, MP3, OGG)
 * - Audio transforms (Resample, ToMono, Normalize, TrimSilence)
 * - Edge cases and error handling
 */

#include <gtest/gtest.h>
#include "../src/decode/audio_decoder.hpp"
#include <cmath>
#include <random>
#include <numeric>

using namespace turboloader::audio;

// ============================================================================
// Helper functions to create test WAV data
// ============================================================================

// Create a valid WAV file header
std::vector<uint8_t> create_wav_header(
    uint32_t data_size,
    uint16_t channels,
    uint32_t sample_rate,
    uint16_t bits_per_sample,
    uint16_t audio_format = 1  // 1 = PCM, 3 = IEEE float
) {
    std::vector<uint8_t> header(44);

    // RIFF header
    header[0] = 'R'; header[1] = 'I'; header[2] = 'F'; header[3] = 'F';
    uint32_t file_size = 36 + data_size;
    header[4] = file_size & 0xFF;
    header[5] = (file_size >> 8) & 0xFF;
    header[6] = (file_size >> 16) & 0xFF;
    header[7] = (file_size >> 24) & 0xFF;

    // WAVE
    header[8] = 'W'; header[9] = 'A'; header[10] = 'V'; header[11] = 'E';

    // fmt chunk
    header[12] = 'f'; header[13] = 'm'; header[14] = 't'; header[15] = ' ';
    header[16] = 16; header[17] = 0; header[18] = 0; header[19] = 0;  // chunk size
    header[20] = audio_format & 0xFF;
    header[21] = (audio_format >> 8) & 0xFF;  // audio format
    header[22] = channels & 0xFF;
    header[23] = (channels >> 8) & 0xFF;  // channels
    header[24] = sample_rate & 0xFF;
    header[25] = (sample_rate >> 8) & 0xFF;
    header[26] = (sample_rate >> 16) & 0xFF;
    header[27] = (sample_rate >> 24) & 0xFF;  // sample rate

    uint32_t byte_rate = sample_rate * channels * bits_per_sample / 8;
    header[28] = byte_rate & 0xFF;
    header[29] = (byte_rate >> 8) & 0xFF;
    header[30] = (byte_rate >> 16) & 0xFF;
    header[31] = (byte_rate >> 24) & 0xFF;  // byte rate

    uint16_t block_align = channels * bits_per_sample / 8;
    header[32] = block_align & 0xFF;
    header[33] = (block_align >> 8) & 0xFF;  // block align
    header[34] = bits_per_sample & 0xFF;
    header[35] = (bits_per_sample >> 8) & 0xFF;  // bits per sample

    // data chunk
    header[36] = 'd'; header[37] = 'a'; header[38] = 't'; header[39] = 'a';
    header[40] = data_size & 0xFF;
    header[41] = (data_size >> 8) & 0xFF;
    header[42] = (data_size >> 16) & 0xFF;
    header[43] = (data_size >> 24) & 0xFF;

    return header;
}

// Create a 16-bit PCM sine wave
std::vector<uint8_t> create_16bit_sine_wav(
    float frequency,
    float duration_sec,
    uint32_t sample_rate = 44100,
    uint16_t channels = 1
) {
    size_t num_samples = static_cast<size_t>(duration_sec * sample_rate);
    size_t data_size = num_samples * channels * 2;

    auto header = create_wav_header(data_size, channels, sample_rate, 16);
    std::vector<uint8_t> wav_data = header;
    wav_data.resize(44 + data_size);

    for (size_t i = 0; i < num_samples; ++i) {
        float t = static_cast<float>(i) / sample_rate;
        float value = std::sin(2.0f * M_PI * frequency * t);
        int16_t sample = static_cast<int16_t>(value * 32767.0f);

        for (int c = 0; c < channels; ++c) {
            size_t offset = 44 + (i * channels + c) * 2;
            wav_data[offset] = sample & 0xFF;
            wav_data[offset + 1] = (sample >> 8) & 0xFF;
        }
    }

    return wav_data;
}

// Create an 8-bit PCM WAV file
std::vector<uint8_t> create_8bit_wav(size_t num_samples, uint32_t sample_rate = 44100) {
    size_t data_size = num_samples;
    auto header = create_wav_header(data_size, 1, sample_rate, 8);
    std::vector<uint8_t> wav_data = header;
    wav_data.resize(44 + data_size);

    for (size_t i = 0; i < num_samples; ++i) {
        // Sawtooth wave
        uint8_t sample = static_cast<uint8_t>((i % 256));
        wav_data[44 + i] = sample;
    }

    return wav_data;
}

// Create a 24-bit PCM WAV file
std::vector<uint8_t> create_24bit_wav(size_t num_samples, uint32_t sample_rate = 44100) {
    size_t data_size = num_samples * 3;
    auto header = create_wav_header(data_size, 1, sample_rate, 24);
    std::vector<uint8_t> wav_data = header;
    wav_data.resize(44 + data_size);

    for (size_t i = 0; i < num_samples; ++i) {
        float t = static_cast<float>(i) / sample_rate;
        float value = std::sin(2.0f * M_PI * 440.0f * t);
        int32_t sample = static_cast<int32_t>(value * 8388607.0f);

        size_t offset = 44 + i * 3;
        wav_data[offset] = sample & 0xFF;
        wav_data[offset + 1] = (sample >> 8) & 0xFF;
        wav_data[offset + 2] = (sample >> 16) & 0xFF;
    }

    return wav_data;
}

// Create a 32-bit IEEE float WAV file
std::vector<uint8_t> create_float_wav(size_t num_samples, uint32_t sample_rate = 44100) {
    size_t data_size = num_samples * 4;
    auto header = create_wav_header(data_size, 1, sample_rate, 32, 3);  // format 3 = IEEE float
    std::vector<uint8_t> wav_data = header;
    wav_data.resize(44 + data_size);

    for (size_t i = 0; i < num_samples; ++i) {
        float t = static_cast<float>(i) / sample_rate;
        float value = std::sin(2.0f * M_PI * 440.0f * t);

        uint8_t* bytes = reinterpret_cast<uint8_t*>(&value);
        size_t offset = 44 + i * 4;
        std::memcpy(&wav_data[offset], bytes, 4);
    }

    return wav_data;
}

// ============================================================================
// Format Detection Tests
// ============================================================================

TEST(AudioFormatDetection, DetectWav) {
    auto wav_data = create_16bit_sine_wav(440.0f, 0.1f);
    EXPECT_EQ(detect_format(wav_data.data(), wav_data.size()), AudioFormat::WAV);
}

TEST(AudioFormatDetection, DetectFlac) {
    // FLAC magic bytes: "fLaC"
    std::vector<uint8_t> flac_data = {'f', 'L', 'a', 'C', 0, 0, 0, 0, 0, 0, 0, 0};
    EXPECT_EQ(detect_format(flac_data.data(), flac_data.size()), AudioFormat::FLAC);
}

TEST(AudioFormatDetection, DetectMp3WithId3) {
    // MP3 with ID3v2 tag
    std::vector<uint8_t> mp3_data = {'I', 'D', '3', 0, 0, 0, 0, 0, 0, 0, 0, 0};
    EXPECT_EQ(detect_format(mp3_data.data(), mp3_data.size()), AudioFormat::MP3);
}

TEST(AudioFormatDetection, DetectMp3FrameSync) {
    // MP3 frame sync
    std::vector<uint8_t> mp3_data = {0xFF, 0xFB, 0x90, 0x00, 0, 0, 0, 0, 0, 0, 0, 0};
    EXPECT_EQ(detect_format(mp3_data.data(), mp3_data.size()), AudioFormat::MP3);
}

TEST(AudioFormatDetection, DetectOgg) {
    // OGG magic bytes: "OggS"
    std::vector<uint8_t> ogg_data = {'O', 'g', 'g', 'S', 0, 0, 0, 0, 0, 0, 0, 0};
    EXPECT_EQ(detect_format(ogg_data.data(), ogg_data.size()), AudioFormat::OGG);
}

TEST(AudioFormatDetection, UnknownFormat) {
    std::vector<uint8_t> unknown = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    EXPECT_EQ(detect_format(unknown.data(), unknown.size()), AudioFormat::UNKNOWN);
}

TEST(AudioFormatDetection, TooSmall) {
    std::vector<uint8_t> small = {0, 0, 0, 0};
    EXPECT_EQ(detect_format(small.data(), small.size()), AudioFormat::UNKNOWN);
}

// ============================================================================
// WAV Decoder Tests
// ============================================================================

TEST(WavDecoder, Decode16BitMono) {
    auto wav_data = create_16bit_sine_wav(440.0f, 0.1f, 44100, 1);

    WavDecoder decoder;
    auto result = decoder.decode(wav_data.data(), wav_data.size());

    EXPECT_TRUE(result.is_success());
    EXPECT_EQ(result.sample_rate, 44100);
    EXPECT_EQ(result.channels, 1);
    EXPECT_EQ(result.bit_depth, 16);
    EXPECT_NEAR(result.duration_seconds, 0.1, 0.01);
    EXPECT_GT(result.samples.size(), 0u);

    // Verify samples are in [-1, 1] range
    for (float sample : result.samples) {
        EXPECT_GE(sample, -1.0f);
        EXPECT_LE(sample, 1.0f);
    }
}

TEST(WavDecoder, Decode16BitStereo) {
    auto wav_data = create_16bit_sine_wav(440.0f, 0.1f, 44100, 2);

    WavDecoder decoder;
    auto result = decoder.decode(wav_data.data(), wav_data.size());

    EXPECT_TRUE(result.is_success());
    EXPECT_EQ(result.channels, 2);
    EXPECT_EQ(result.samples.size(), result.num_samples() * 2);
}

TEST(WavDecoder, Decode8Bit) {
    auto wav_data = create_8bit_wav(4410, 44100);

    WavDecoder decoder;
    auto result = decoder.decode(wav_data.data(), wav_data.size());

    EXPECT_TRUE(result.is_success());
    EXPECT_EQ(result.bit_depth, 8);
    EXPECT_EQ(result.samples.size(), 4410u);

    // 8-bit samples should be in [-1, 1] range after conversion
    for (float sample : result.samples) {
        EXPECT_GE(sample, -1.0f);
        EXPECT_LE(sample, 1.0f);
    }
}

TEST(WavDecoder, Decode24Bit) {
    auto wav_data = create_24bit_wav(4410, 44100);

    WavDecoder decoder;
    auto result = decoder.decode(wav_data.data(), wav_data.size());

    EXPECT_TRUE(result.is_success());
    EXPECT_EQ(result.bit_depth, 24);
    EXPECT_GT(result.samples.size(), 0u);
}

TEST(WavDecoder, DecodeIEEEFloat) {
    auto wav_data = create_float_wav(4410, 44100);

    WavDecoder decoder;
    auto result = decoder.decode(wav_data.data(), wav_data.size());

    EXPECT_TRUE(result.is_success());
    EXPECT_EQ(result.bit_depth, 32);
    EXPECT_EQ(result.samples.size(), 4410u);

    // Check that we get the original sine wave back
    for (size_t i = 0; i < 100; ++i) {
        float t = static_cast<float>(i) / 44100.0f;
        float expected = std::sin(2.0f * M_PI * 440.0f * t);
        EXPECT_NEAR(result.samples[i], expected, 0.001f);
    }
}

TEST(WavDecoder, DifferentSampleRates) {
    std::vector<uint32_t> sample_rates = {8000, 16000, 22050, 44100, 48000, 96000};

    for (uint32_t rate : sample_rates) {
        auto wav_data = create_16bit_sine_wav(440.0f, 0.01f, rate, 1);

        WavDecoder decoder;
        auto result = decoder.decode(wav_data.data(), wav_data.size());

        EXPECT_TRUE(result.is_success()) << "Failed for sample rate " << rate;
        EXPECT_EQ(result.sample_rate, static_cast<int>(rate));
    }
}

TEST(WavDecoder, InvalidRiffHeader) {
    std::vector<uint8_t> invalid = {'X', 'I', 'F', 'F', 0, 0, 0, 0, 'W', 'A', 'V', 'E'};
    invalid.resize(44);

    WavDecoder decoder;
    auto result = decoder.decode(invalid.data(), invalid.size());

    EXPECT_FALSE(result.is_success());
    EXPECT_FALSE(result.error_message.empty());
}

TEST(WavDecoder, InvalidWaveHeader) {
    std::vector<uint8_t> invalid = {'R', 'I', 'F', 'F', 0, 0, 0, 0, 'X', 'A', 'V', 'E'};
    invalid.resize(44);

    WavDecoder decoder;
    auto result = decoder.decode(invalid.data(), invalid.size());

    EXPECT_FALSE(result.is_success());
}

TEST(WavDecoder, TooSmall) {
    std::vector<uint8_t> small(10);

    WavDecoder decoder;
    auto result = decoder.decode(small.data(), small.size());

    EXPECT_FALSE(result.is_success());
}

// ============================================================================
// Audio Decoder (Unified) Tests
// ============================================================================

TEST(AudioDecoder, AutoDetectWav) {
    auto wav_data = create_16bit_sine_wav(440.0f, 0.1f);

    AudioDecoder decoder;
    auto result = decoder.decode(wav_data.data(), wav_data.size());

    EXPECT_TRUE(result.is_success());
    EXPECT_EQ(result.format, "WAV");
}

TEST(AudioDecoder, ExplicitFormat) {
    auto wav_data = create_16bit_sine_wav(440.0f, 0.1f);

    AudioDecoder decoder;
    auto result = decoder.decode(wav_data.data(), wav_data.size(), AudioFormat::WAV);

    EXPECT_TRUE(result.is_success());
}

TEST(AudioDecoder, UnknownFormat) {
    std::vector<uint8_t> unknown(100, 0);

    AudioDecoder decoder;
    auto result = decoder.decode(unknown.data(), unknown.size());

    EXPECT_FALSE(result.is_success());
    EXPECT_FALSE(result.error_message.empty());
}

TEST(AudioDecoder, SupportedFormats) {
    std::string formats = AudioDecoder::supported_formats();
    EXPECT_FALSE(formats.empty());
    EXPECT_NE(formats.find("WAV"), std::string::npos);
}

TEST(AudioDecoder, IsFormatSupported) {
    EXPECT_TRUE(AudioDecoder::is_format_supported(AudioFormat::WAV));
    EXPECT_FALSE(AudioDecoder::is_format_supported(AudioFormat::UNKNOWN));
}

// ============================================================================
// Audio Transform Tests
// ============================================================================

TEST(AudioTransforms, ResampleUpSample) {
    auto wav_data = create_16bit_sine_wav(440.0f, 0.1f, 22050, 1);

    WavDecoder decoder;
    auto input = decoder.decode(wav_data.data(), wav_data.size());
    ASSERT_TRUE(input.is_success());

    Resample resample(44100);
    auto output = resample.apply(input);

    EXPECT_EQ(output.sample_rate, 44100);
    EXPECT_NEAR(output.duration_seconds, input.duration_seconds, 0.01);
    // Upsampling should roughly double the samples
    EXPECT_NEAR(output.samples.size(), input.samples.size() * 2, input.samples.size() * 0.1);
}

TEST(AudioTransforms, ResampleDownSample) {
    auto wav_data = create_16bit_sine_wav(440.0f, 0.1f, 44100, 1);

    WavDecoder decoder;
    auto input = decoder.decode(wav_data.data(), wav_data.size());
    ASSERT_TRUE(input.is_success());

    Resample resample(22050);
    auto output = resample.apply(input);

    EXPECT_EQ(output.sample_rate, 22050);
    // Downsampling should roughly halve the samples
    EXPECT_NEAR(output.samples.size(), input.samples.size() / 2, input.samples.size() * 0.1);
}

TEST(AudioTransforms, ResampleSameRate) {
    auto wav_data = create_16bit_sine_wav(440.0f, 0.1f, 44100, 1);

    WavDecoder decoder;
    auto input = decoder.decode(wav_data.data(), wav_data.size());
    ASSERT_TRUE(input.is_success());

    Resample resample(44100);
    auto output = resample.apply(input);

    // Should be unchanged
    EXPECT_EQ(output.samples.size(), input.samples.size());
}

TEST(AudioTransforms, ToMonoFromStereo) {
    auto wav_data = create_16bit_sine_wav(440.0f, 0.1f, 44100, 2);

    WavDecoder decoder;
    auto input = decoder.decode(wav_data.data(), wav_data.size());
    ASSERT_TRUE(input.is_success());
    EXPECT_EQ(input.channels, 2);

    ToMono to_mono;
    auto output = to_mono.apply(input);

    EXPECT_EQ(output.channels, 1);
    EXPECT_EQ(output.samples.size(), input.num_samples());
}

TEST(AudioTransforms, ToMonoAlreadyMono) {
    auto wav_data = create_16bit_sine_wav(440.0f, 0.1f, 44100, 1);

    WavDecoder decoder;
    auto input = decoder.decode(wav_data.data(), wav_data.size());
    ASSERT_TRUE(input.is_success());

    ToMono to_mono;
    auto output = to_mono.apply(input);

    EXPECT_EQ(output.channels, 1);
    EXPECT_EQ(output.samples.size(), input.samples.size());
}

TEST(AudioTransforms, Normalize) {
    auto wav_data = create_16bit_sine_wav(440.0f, 0.1f, 44100, 1);

    WavDecoder decoder;
    auto input = decoder.decode(wav_data.data(), wav_data.size());
    ASSERT_TRUE(input.is_success());

    // Scale down the input
    for (float& s : input.samples) {
        s *= 0.1f;
    }

    float peak_before = 0.0f;
    for (float s : input.samples) {
        peak_before = std::max(peak_before, std::abs(s));
    }

    Normalize normalize(-3.0f);  // Normalize to -3dB
    auto output = normalize.apply(input);

    float peak_after = 0.0f;
    for (float s : output.samples) {
        peak_after = std::max(peak_after, std::abs(s));
    }

    // Peak should be closer to target
    float target = std::pow(10.0f, -3.0f / 20.0f);
    EXPECT_NEAR(peak_after, target, 0.01f);
}

TEST(AudioTransforms, NormalizeSilent) {
    AudioResult silent;
    silent.sample_rate = 44100;
    silent.channels = 1;
    silent.bit_depth = 16;
    silent.samples.resize(1000, 0.0f);

    Normalize normalize;
    auto output = normalize.apply(silent);

    // Should not change silent audio
    for (float s : output.samples) {
        EXPECT_FLOAT_EQ(s, 0.0f);
    }
}

TEST(AudioTransforms, TrimSilence) {
    // Create audio with silence at beginning and end
    AudioResult input;
    input.sample_rate = 44100;
    input.channels = 1;
    input.bit_depth = 16;
    input.samples.resize(10000, 0.0f);

    // Add non-silent section in the middle
    for (size_t i = 2000; i < 8000; ++i) {
        float t = static_cast<float>(i) / 44100.0f;
        input.samples[i] = std::sin(2.0f * M_PI * 440.0f * t) * 0.5f;
    }

    TrimSilence trim(-60.0f);  // -60dB threshold
    auto output = trim.apply(input);

    // Should be trimmed to just the non-silent section
    EXPECT_LT(output.samples.size(), input.samples.size());
    EXPECT_NEAR(output.samples.size(), 6000u, 100u);
}

TEST(AudioTransforms, TrimSilenceAllSilent) {
    AudioResult silent;
    silent.sample_rate = 44100;
    silent.channels = 1;
    silent.bit_depth = 16;
    silent.samples.resize(1000, 0.0f);

    TrimSilence trim;
    auto output = trim.apply(silent);

    // Empty audio should remain
    EXPECT_EQ(output.samples.size(), 0u);
}

TEST(AudioTransforms, ChainedTransforms) {
    auto wav_data = create_16bit_sine_wav(440.0f, 0.2f, 44100, 2);

    WavDecoder decoder;
    auto result = decoder.decode(wav_data.data(), wav_data.size());
    ASSERT_TRUE(result.is_success());

    // Chain: stereo->mono, then resample, then normalize
    ToMono to_mono;
    Resample resample(22050);
    Normalize normalize(-6.0f);

    auto mono = to_mono.apply(result);
    auto resampled = resample.apply(mono);
    auto normalized = normalize.apply(resampled);

    EXPECT_EQ(normalized.channels, 1);
    EXPECT_EQ(normalized.sample_rate, 22050);

    float peak = 0.0f;
    for (float s : normalized.samples) {
        peak = std::max(peak, std::abs(s));
    }
    float target = std::pow(10.0f, -6.0f / 20.0f);
    EXPECT_NEAR(peak, target, 0.01f);
}

// ============================================================================
// AudioResult Tests
// ============================================================================

TEST(AudioResult, IsSuccess) {
    AudioResult success;
    success.samples = {0.0f, 0.1f, 0.2f};
    success.sample_rate = 44100;
    success.channels = 1;
    EXPECT_TRUE(success.is_success());

    AudioResult empty_samples;
    empty_samples.sample_rate = 44100;
    EXPECT_FALSE(empty_samples.is_success());

    AudioResult with_error;
    with_error.samples = {0.0f};
    with_error.error_message = "Some error";
    EXPECT_FALSE(with_error.is_success());
}

TEST(AudioResult, NumSamples) {
    AudioResult mono;
    mono.samples = {0.0f, 0.1f, 0.2f, 0.3f};
    mono.channels = 1;
    EXPECT_EQ(mono.num_samples(), 4u);

    AudioResult stereo;
    stereo.samples = {0.0f, 0.1f, 0.2f, 0.3f};
    stereo.channels = 2;
    EXPECT_EQ(stereo.num_samples(), 2u);
}

// ============================================================================
// Performance/Stress Tests
// ============================================================================

TEST(AudioPerformance, DecodeLongFile) {
    // Create 10 seconds of audio
    auto wav_data = create_16bit_sine_wav(440.0f, 10.0f, 44100, 2);

    WavDecoder decoder;
    auto result = decoder.decode(wav_data.data(), wav_data.size());

    EXPECT_TRUE(result.is_success());
    EXPECT_NEAR(result.duration_seconds, 10.0, 0.1);

    // 10 seconds * 44100 Hz * 2 channels
    EXPECT_NEAR(result.samples.size(), 10 * 44100 * 2, 1000);
}

TEST(AudioPerformance, MultipleDecodes) {
    auto wav_data = create_16bit_sine_wav(440.0f, 0.1f);

    WavDecoder decoder;

    for (int i = 0; i < 100; ++i) {
        auto result = decoder.decode(wav_data.data(), wav_data.size());
        EXPECT_TRUE(result.is_success());
    }
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
