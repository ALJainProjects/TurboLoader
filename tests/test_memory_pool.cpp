#include "turboloader/core/memory_pool.hpp"
#include <gtest/gtest.h>

using namespace turboloader;

TEST(MemoryPoolTest, BasicAllocation) {
    MemoryPool pool(1024);

    void* ptr1 = pool.allocate(64);
    ASSERT_NE(ptr1, nullptr);

    void* ptr2 = pool.allocate(128);
    ASSERT_NE(ptr2, nullptr);

    // Pointers should be different
    EXPECT_NE(ptr1, ptr2);
}

TEST(MemoryPoolTest, Alignment) {
    MemoryPool pool(1024, 64);

    void* ptr = pool.allocate(10, 64);
    ASSERT_NE(ptr, nullptr);

    // Check 64-byte alignment
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    EXPECT_EQ(addr % 64, 0);
}

TEST(MemoryPoolTest, Reset) {
    MemoryPool pool(1024);

    void* ptr1 = pool.allocate(256);
    ASSERT_NE(ptr1, nullptr);

    size_t used = pool.used_bytes();
    EXPECT_GT(used, 0);

    pool.reset();
    EXPECT_EQ(pool.used_bytes(), 0);

    // Can allocate again after reset
    void* ptr2 = pool.allocate(256);
    ASSERT_NE(ptr2, nullptr);
}

TEST(MemoryPoolTest, LargeAllocation) {
    MemoryPool pool(1024);  // 1KB blocks

    // Allocate 2KB (larger than block size)
    void* ptr = pool.allocate(2048);
    ASSERT_NE(ptr, nullptr);

    EXPECT_GT(pool.total_bytes(), 2048);
}
