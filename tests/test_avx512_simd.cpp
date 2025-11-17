/**
 * @file test_avx512_simd.cpp
 * @brief Unit tests for AVX-512 SIMD optimizations
 *
 * Tests correctness and performance of AVX-512 vectorized operations.
 */

#include "../src/transforms/simd_utils.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>

using namespace turboloader::transforms::simd;

// Test parameters
constexpr size_t TEST_SIZE = 1024 * 1024; // 1M elements for performance testing
constexpr float TOLERANCE = 1e-5f;

// Helper function to check if two floats are approximately equal
bool approx_equal(float a, float b, float tol = TOLERANCE) {
    return std::abs(a - b) < tol;
}

// Helper function to print test result
void print_result(const char* test_name, bool passed) {
    std::cout << "[" << (passed ? "PASS" : "FAIL") << "] " << test_name << std::endl;
}

// Test 1: uint8 to float32 conversion
bool test_cvt_u8_to_f32() {
    std::vector<uint8_t> input(TEST_SIZE);
    std::vector<float> output(TEST_SIZE);

    // Fill with test pattern
    for (size_t i = 0; i < TEST_SIZE; ++i) {
        input[i] = static_cast<uint8_t>(i % 256);
    }

    // Convert
    cvt_u8_to_f32_normalized(input.data(), output.data(), TEST_SIZE);

    // Verify
    bool passed = true;
    for (size_t i = 0; i < TEST_SIZE; ++i) {
        float expected = input[i] / 255.0f;
        if (!approx_equal(output[i], expected)) {
            std::cerr << "  Mismatch at index " << i << ": "
                      << "expected " << expected << ", got " << output[i] << std::endl;
            passed = false;
            break;
        }
    }

    return passed;
}

// Test 2: float32 to uint8 conversion
bool test_cvt_f32_to_u8() {
    std::vector<float> input(TEST_SIZE);
    std::vector<uint8_t> output(TEST_SIZE);

    // Fill with test pattern (normalized values)
    for (size_t i = 0; i < TEST_SIZE; ++i) {
        input[i] = (i % 256) / 255.0f;
    }

    // Convert
    cvt_f32_to_u8_clamped(input.data(), output.data(), TEST_SIZE);

    // Verify
    bool passed = true;
    for (size_t i = 0; i < TEST_SIZE; ++i) {
        uint8_t expected = static_cast<uint8_t>(input[i] * 255.0f + 0.5f);
        if (output[i] != expected) {
            std::cerr << "  Mismatch at index " << i << ": "
                      << "expected " << static_cast<int>(expected)
                      << ", got " << static_cast<int>(output[i]) << std::endl;
            passed = false;
            break;
        }
    }

    return passed;
}

// Test 3: uint8 multiply by scalar
bool test_mul_u8_scalar() {
    std::vector<uint8_t> input(TEST_SIZE);
    std::vector<uint8_t> output(TEST_SIZE);
    const float scalar = 1.5f;

    // Fill with test pattern
    for (size_t i = 0; i < TEST_SIZE; ++i) {
        input[i] = static_cast<uint8_t>(std::min(170ul, i % 256)); // Avoid overflow
    }

    // Multiply
    mul_u8_scalar(input.data(), output.data(), scalar, TEST_SIZE);

    // Verify
    bool passed = true;
    for (size_t i = 0; i < TEST_SIZE; ++i) {
        float expected_f = std::min(255.0f, input[i] * scalar);
        uint8_t expected = static_cast<uint8_t>(expected_f);
        if (output[i] != expected) {
            std::cerr << "  Mismatch at index " << i << ": "
                      << "expected " << static_cast<int>(expected)
                      << ", got " << static_cast<int>(output[i]) << std::endl;
            passed = false;
            break;
        }
    }

    return passed;
}

// Test 4: uint8 add scalar
bool test_add_u8_scalar() {
    std::vector<uint8_t> input(TEST_SIZE);
    std::vector<uint8_t> output(TEST_SIZE);
    const float scalar = 50.0f;

    // Fill with test pattern
    for (size_t i = 0; i < TEST_SIZE; ++i) {
        input[i] = static_cast<uint8_t>(std::min(200ul, i % 256)); // Avoid overflow
    }

    // Add
    add_u8_scalar(input.data(), output.data(), scalar, TEST_SIZE);

    // Verify
    bool passed = true;
    for (size_t i = 0; i < TEST_SIZE; ++i) {
        float expected_f = std::min(255.0f, input[i] + scalar);
        uint8_t expected = static_cast<uint8_t>(expected_f);
        if (output[i] != expected) {
            std::cerr << "  Mismatch at index " << i << ": "
                      << "expected " << static_cast<int>(expected)
                      << ", got " << static_cast<int>(output[i]) << std::endl;
            passed = false;
            break;
        }
    }

    return passed;
}

// Test 5: float32 normalize
bool test_normalize_f32() {
    std::vector<float> input(TEST_SIZE);
    std::vector<float> output(TEST_SIZE);
    const float mean = 0.5f;
    const float std = 0.25f;

    // Fill with test pattern
    for (size_t i = 0; i < TEST_SIZE; ++i) {
        input[i] = (i % 1000) / 1000.0f;
    }

    // Normalize
    normalize_f32(input.data(), output.data(), mean, std, TEST_SIZE);

    // Verify
    bool passed = true;
    for (size_t i = 0; i < TEST_SIZE; ++i) {
        float expected = (input[i] - mean) / std;
        if (!approx_equal(output[i], expected, 1e-4f)) {
            std::cerr << "  Mismatch at index " << i << ": "
                      << "expected " << expected << ", got " << output[i] << std::endl;
            passed = false;
            break;
        }
    }

    return passed;
}

// Performance benchmark
void benchmark_simd_operations() {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "AVX-512 SIMD Performance Benchmarks" << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    const size_t BENCH_SIZE = 10 * 1024 * 1024; // 10M elements
    const int ITERATIONS = 100;

    std::vector<uint8_t> u8_data(BENCH_SIZE);
    std::vector<float> f32_data(BENCH_SIZE);
    std::vector<uint8_t> u8_output(BENCH_SIZE);
    std::vector<float> f32_output(BENCH_SIZE);

    // Fill with random-ish data
    for (size_t i = 0; i < BENCH_SIZE; ++i) {
        u8_data[i] = static_cast<uint8_t>(i % 256);
        f32_data[i] = (i % 1000) / 1000.0f;
    }

    // Benchmark 1: uint8 to float32 conversion
    {
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < ITERATIONS; ++i) {
            cvt_u8_to_f32_normalized(u8_data.data(), f32_output.data(), BENCH_SIZE);
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        double throughput = (BENCH_SIZE * ITERATIONS) / (duration.count() / 1e6) / 1e6;
        std::cout << "cvt_u8_to_f32_normalized:  " << std::fixed << std::setprecision(2)
                  << throughput << " M elements/s" << std::endl;
    }

    // Benchmark 2: float32 to uint8 conversion
    {
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < ITERATIONS; ++i) {
            cvt_f32_to_u8_clamped(f32_data.data(), u8_output.data(), BENCH_SIZE);
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        double throughput = (BENCH_SIZE * ITERATIONS) / (duration.count() / 1e6) / 1e6;
        std::cout << "cvt_f32_to_u8_clamped:     " << std::fixed << std::setprecision(2)
                  << throughput << " M elements/s" << std::endl;
    }

    // Benchmark 3: multiply uint8 by scalar
    {
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < ITERATIONS; ++i) {
            mul_u8_scalar(u8_data.data(), u8_output.data(), 1.5f, BENCH_SIZE);
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        double throughput = (BENCH_SIZE * ITERATIONS) / (duration.count() / 1e6) / 1e6;
        std::cout << "mul_u8_scalar:             " << std::fixed << std::setprecision(2)
                  << throughput << " M elements/s" << std::endl;
    }

    // Benchmark 4: add scalar to uint8
    {
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < ITERATIONS; ++i) {
            add_u8_scalar(u8_data.data(), u8_output.data(), 50.0f, BENCH_SIZE);
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        double throughput = (BENCH_SIZE * ITERATIONS) / (duration.count() / 1e6) / 1e6;
        std::cout << "add_u8_scalar:             " << std::fixed << std::setprecision(2)
                  << throughput << " M elements/s" << std::endl;
    }

    // Benchmark 5: normalize float32
    {
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < ITERATIONS; ++i) {
            normalize_f32(f32_data.data(), f32_output.data(), 0.5f, 0.25f, BENCH_SIZE);
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        double throughput = (BENCH_SIZE * ITERATIONS) / (duration.count() / 1e6) / 1e6;
        std::cout << "normalize_f32:             " << std::fixed << std::setprecision(2)
                  << throughput << " M elements/s" << std::endl;
    }

    std::cout << std::string(80, '=') << std::endl;
}

int main() {
    std::cout << "================================================================================" << std::endl;
    std::cout << "AVX-512 SIMD Unit Tests" << std::endl;
    std::cout << "================================================================================" << std::endl;

    // Check SIMD availability
#ifdef TURBOLOADER_SIMD_AVX512
    std::cout << "SIMD Mode: AVX-512 (64-byte vectors, 16-wide float)" << std::endl;
#elif defined(TURBOLOADER_SIMD_AVX2)
    std::cout << "SIMD Mode: AVX2 (32-byte vectors, 8-wide float)" << std::endl;
    std::cout << "WARNING: AVX-512 not available! Tests will use AVX2 fallback." << std::endl;
#elif defined(TURBOLOADER_SIMD_NEON)
    std::cout << "SIMD Mode: NEON (16-byte vectors, 4-wide float)" << std::endl;
    std::cout << "WARNING: AVX-512 not available! Tests will use NEON fallback." << std::endl;
#else
    std::cout << "SIMD Mode: Scalar (no vectorization)" << std::endl;
    std::cout << "WARNING: AVX-512 not available! Tests will use scalar fallback." << std::endl;
#endif

    std::cout << std::endl;

    // Run correctness tests
    int passed = 0;
    int total = 5;

    if (test_cvt_u8_to_f32()) passed++;
    print_result("cvt_u8_to_f32_normalized", test_cvt_u8_to_f32());

    if (test_cvt_f32_to_u8()) passed++;
    print_result("cvt_f32_to_u8_clamped", test_cvt_f32_to_u8());

    if (test_mul_u8_scalar()) passed++;
    print_result("mul_u8_scalar", test_mul_u8_scalar());

    if (test_add_u8_scalar()) passed++;
    print_result("add_u8_scalar", test_add_u8_scalar());

    if (test_normalize_f32()) passed++;
    print_result("normalize_f32", test_normalize_f32());

    std::cout << "\nTest Summary: " << passed << "/" << total << " tests passed" << std::endl;

    // Run performance benchmarks
    benchmark_simd_operations();

    return (passed == total) ? 0 : 1;
}
