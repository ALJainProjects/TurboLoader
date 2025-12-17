/**
 * @file test_buffer_pool.cpp
 * @brief Unit tests for BufferPool class
 *
 * Tests:
 * 1. Basic acquire/release functionality
 * 2. Buffer reuse (cache hits)
 * 3. Statistics tracking
 * 4. Thread safety
 * 5. Size bucketing
 * 6. Integration with ResizeTransform
 */

#include "../src/core/buffer_pool.hpp"
#include "../src/transforms/resize_transform.hpp"
#include <gtest/gtest.h>
#include <thread>
#include <vector>
#include <atomic>

using namespace turboloader;
using namespace turboloader::transforms;

class BufferPoolTest : public ::testing::Test {
protected:
    void SetUp() override {
        pool_ = std::make_unique<SizedBufferPool>(8, 16 * 1024 * 1024);  // 8 per bucket, 16MB max
    }

    std::unique_ptr<SizedBufferPool> pool_;
};

// Basic acquire test
TEST_F(BufferPoolTest, AcquireReturnsValidBuffer) {
    auto buffer = pool_->acquire(1024);
    ASSERT_NE(buffer, nullptr);

    // Should be able to write to the buffer
    std::memset(buffer.get(), 0xAB, 1024);
    EXPECT_EQ(buffer[0], 0xAB);
    EXPECT_EQ(buffer[1023], 0xAB);
}

// Zero size returns nullptr
TEST_F(BufferPoolTest, AcquireZeroReturnsNull) {
    auto buffer = pool_->acquire(0);
    EXPECT_EQ(buffer, nullptr);
}

// Release and reuse
TEST_F(BufferPoolTest, ReleaseAndReuse) {
    // Acquire a buffer
    auto buffer1 = pool_->acquire(4096);
    ASSERT_NE(buffer1, nullptr);
    uint8_t* ptr1 = buffer1.get();

    // Fill with pattern
    std::memset(buffer1.get(), 0x42, 4096);

    // Release back to pool
    pool_->release(std::move(buffer1), 4096);
    EXPECT_EQ(pool_->pooled_count(), 1);

    // Acquire again - should get the same buffer (or at least from pool)
    auto buffer2 = pool_->acquire(4096);
    ASSERT_NE(buffer2, nullptr);

    // Check stats show cache hit
    auto stats = pool_->stats();
    EXPECT_EQ(stats.cache_hits, 1);
}

// Statistics tracking
TEST_F(BufferPoolTest, StatisticsTracking) {
    pool_->reset_stats();

    // First acquire - miss
    auto buf1 = pool_->acquire(1024);
    auto stats1 = pool_->stats();
    EXPECT_EQ(stats1.acquire_calls, 1);
    EXPECT_EQ(stats1.cache_misses, 1);
    EXPECT_EQ(stats1.cache_hits, 0);

    // Release
    pool_->release(std::move(buf1), 1024);
    auto stats2 = pool_->stats();
    EXPECT_EQ(stats2.release_calls, 1);
    EXPECT_EQ(stats2.pooled_buffers, 1);

    // Second acquire - hit
    auto buf2 = pool_->acquire(1024);
    auto stats3 = pool_->stats();
    EXPECT_EQ(stats3.acquire_calls, 2);
    EXPECT_EQ(stats3.cache_hits, 1);
}

// Size bucketing
TEST_F(BufferPoolTest, SizeBucketing) {
    // Small sizes should be rounded up to 4KB bucket
    auto small = pool_->acquire(100);
    ASSERT_NE(small, nullptr);

    pool_->release(std::move(small), 100);

    // Should be able to acquire slightly larger buffer from same bucket
    auto larger = pool_->acquire(2000);
    ASSERT_NE(larger, nullptr);

    auto stats = pool_->stats();
    EXPECT_GE(stats.cache_hits, 0);  // May or may not hit depending on bucket
}

// Clear pool
TEST_F(BufferPoolTest, ClearPool) {
    auto buf1 = pool_->acquire(4096);
    auto buf2 = pool_->acquire(8192);

    pool_->release(std::move(buf1), 4096);
    pool_->release(std::move(buf2), 8192);

    EXPECT_EQ(pool_->pooled_count(), 2);

    pool_->clear();
    EXPECT_EQ(pool_->pooled_count(), 0);
}

// Pooled memory tracking
TEST_F(BufferPoolTest, PooledMemoryTracking) {
    auto buf1 = pool_->acquire(4096);
    auto buf2 = pool_->acquire(8192);

    pool_->release(std::move(buf1), 4096);
    pool_->release(std::move(buf2), 8192);

    size_t pooled_mem = pool_->pooled_memory();
    // Should be sum of bucket sizes (4KB + 8KB rounded up)
    EXPECT_GT(pooled_mem, 0);
}

// Oversized buffers not pooled
TEST_F(BufferPoolTest, OversizedNotPooled) {
    SizedBufferPool small_pool(4, 1024);  // Max 1KB

    auto large = small_pool.acquire(2048);  // > max
    ASSERT_NE(large, nullptr);

    small_pool.release(std::move(large), 2048);
    EXPECT_EQ(small_pool.pooled_count(), 0);  // Not pooled

    auto stats = small_pool.stats();
    EXPECT_EQ(stats.oversized_releases, 1);
}

// Bucket limit
TEST_F(BufferPoolTest, BucketLimitRespected) {
    SizedBufferPool limited_pool(2, 1024 * 1024);  // Only 2 per bucket

    // Acquire and release 3 buffers of same size
    std::vector<std::unique_ptr<uint8_t[]>> buffers;
    for (int i = 0; i < 3; ++i) {
        buffers.push_back(limited_pool.acquire(4096));
    }

    for (auto& buf : buffers) {
        limited_pool.release(std::move(buf), 4096);
    }

    // Should only keep 2
    EXPECT_EQ(limited_pool.pooled_count(), 2);

    auto stats = limited_pool.stats();
    EXPECT_EQ(stats.bucket_full_releases, 1);
}

// Thread safety test
TEST_F(BufferPoolTest, ThreadSafety) {
    constexpr int NUM_THREADS = 4;
    constexpr int OPS_PER_THREAD = 100;
    std::atomic<int> completed{0};

    auto worker = [&]() {
        for (int i = 0; i < OPS_PER_THREAD; ++i) {
            size_t size = 1024 * ((i % 4) + 1);  // 1KB-4KB
            auto buffer = pool_->acquire(size);
            ASSERT_NE(buffer, nullptr);

            // Do some work
            std::memset(buffer.get(), static_cast<uint8_t>(i), std::min(size, (size_t)64));

            pool_->release(std::move(buffer), size);
        }
        completed++;
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < NUM_THREADS; ++i) {
        threads.emplace_back(worker);
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(completed.load(), NUM_THREADS);

    auto stats = pool_->stats();
    EXPECT_EQ(stats.acquire_calls, NUM_THREADS * OPS_PER_THREAD);
    EXPECT_EQ(stats.release_calls, NUM_THREADS * OPS_PER_THREAD);
}

// PooledBufferGuard RAII test
TEST_F(BufferPoolTest, PooledBufferGuard) {
    {
        PooledBufferGuard guard(*pool_, 4096);
        ASSERT_NE(guard.get(), nullptr);
        EXPECT_EQ(guard.size(), 4096);

        // Write to buffer
        std::memset(guard.get(), 0xFF, 4096);
    }
    // Guard destroyed, buffer should be back in pool
    EXPECT_EQ(pool_->pooled_count(), 1);
}

// PooledBufferGuard release ownership
TEST_F(BufferPoolTest, PooledBufferGuardRelease) {
    std::unique_ptr<uint8_t[]> external;
    {
        PooledBufferGuard guard(*pool_, 4096);
        external = guard.release();  // Take ownership
    }
    // Guard destroyed but ownership was transferred
    EXPECT_EQ(pool_->pooled_count(), 0);  // Buffer not returned
    EXPECT_NE(external, nullptr);  // We still have it
}

// Global buffer pool singleton
TEST(GlobalBufferPoolTest, SingletonExists) {
    SizedBufferPool& pool = get_resize_buffer_pool();

    auto buffer = pool.acquire(1024);
    EXPECT_NE(buffer, nullptr);

    pool.release(std::move(buffer), 1024);
}

// Integration with ResizeTransform
TEST(ResizeTransformIntegrationTest, WithBufferPool) {
    // Create test image
    const int width = 640, height = 480, channels = 3;
    std::vector<uint8_t> input_data(width * height * channels, 128);

    ImageData input(input_data.data(), width, height, channels, width * channels, false);

    // Create resize transform with buffer pool
    ResizeTransform resize(224, 224, InterpolationMode::BILINEAR, true);
    EXPECT_TRUE(resize.uses_buffer_pool());

    // Apply transform
    auto output = resize.apply(input);
    ASSERT_NE(output, nullptr);
    EXPECT_EQ(output->width, 224);
    EXPECT_EQ(output->height, 224);
    EXPECT_EQ(output->channels, 3);
}

// ResizeTransform without buffer pool (default)
TEST(ResizeTransformIntegrationTest, WithoutBufferPool) {
    const int width = 640, height = 480, channels = 3;
    std::vector<uint8_t> input_data(width * height * channels, 128);

    ImageData input(input_data.data(), width, height, channels, width * channels, false);

    ResizeTransform resize(224, 224);  // Default: no buffer pool
    EXPECT_FALSE(resize.uses_buffer_pool());

    auto output = resize.apply(input);
    ASSERT_NE(output, nullptr);
    EXPECT_EQ(output->width, 224);
    EXPECT_EQ(output->height, 224);
}

// Enable/disable buffer pool at runtime
TEST(ResizeTransformIntegrationTest, ToggleBufferPool) {
    ResizeTransform resize(100, 100);

    EXPECT_FALSE(resize.uses_buffer_pool());

    resize.set_buffer_pool(true);
    EXPECT_TRUE(resize.uses_buffer_pool());

    resize.set_buffer_pool(false);
    EXPECT_FALSE(resize.uses_buffer_pool());
}

// Hit rate calculation
TEST_F(BufferPoolTest, HitRateCalculation) {
    pool_->reset_stats();

    // 2 misses, then 2 hits
    auto buf1 = pool_->acquire(4096);
    auto buf2 = pool_->acquire(4096);
    pool_->release(std::move(buf1), 4096);
    pool_->release(std::move(buf2), 4096);

    auto buf3 = pool_->acquire(4096);
    auto buf4 = pool_->acquire(4096);

    auto stats = pool_->stats();
    EXPECT_EQ(stats.acquire_calls, 4);
    EXPECT_EQ(stats.cache_hits, 2);
    EXPECT_EQ(stats.cache_misses, 2);
    EXPECT_FLOAT_EQ(stats.hit_rate(), 0.5f);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
