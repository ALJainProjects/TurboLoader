#include "turboloader/core/lock_free_queue.hpp"
#include <gtest/gtest.h>
#include <thread>
#include <vector>

using namespace turboloader;

TEST(LockFreeQueueTest, BasicPushPop) {
    LockFreeSPMCQueue<int> queue(16);

    EXPECT_TRUE(queue.empty());
    EXPECT_EQ(queue.size(), 0);

    // Push some items
    EXPECT_TRUE(queue.try_push(1));
    EXPECT_TRUE(queue.try_push(2));
    EXPECT_TRUE(queue.try_push(3));

    EXPECT_FALSE(queue.empty());
    EXPECT_EQ(queue.size(), 3);

    // Pop them
    auto val1 = queue.try_pop();
    ASSERT_TRUE(val1.has_value());
    EXPECT_EQ(*val1, 1);

    auto val2 = queue.try_pop();
    ASSERT_TRUE(val2.has_value());
    EXPECT_EQ(*val2, 2);

    auto val3 = queue.try_pop();
    ASSERT_TRUE(val3.has_value());
    EXPECT_EQ(*val3, 3);

    // Queue should be empty
    EXPECT_TRUE(queue.empty());
    auto val4 = queue.try_pop();
    EXPECT_FALSE(val4.has_value());
}

TEST(LockFreeQueueTest, FullQueue) {
    LockFreeSPMCQueue<int> queue(4);

    // Fill queue
    EXPECT_TRUE(queue.try_push(1));
    EXPECT_TRUE(queue.try_push(2));
    EXPECT_TRUE(queue.try_push(3));
    EXPECT_TRUE(queue.try_push(4));

    // Should be full
    EXPECT_FALSE(queue.try_push(5));
}

TEST(LockFreeQueueTest, MultipleConsumers) {
    LockFreeSPMCQueue<int> queue(1024);

    // Producer thread
    std::thread producer([&]() {
        for (int i = 0; i < 1000; ++i) {
            while (!queue.try_push(i)) {
                std::this_thread::yield();
            }
        }
    });

    // Multiple consumer threads
    std::atomic<int> total{0};
    std::vector<std::thread> consumers;

    for (int i = 0; i < 4; ++i) {
        consumers.emplace_back([&]() {
            int local_sum = 0;
            int count = 0;

            while (count < 250) {  // Each consumer gets ~250 items
                auto val = queue.try_pop();
                if (val) {
                    local_sum += *val;
                    count++;
                } else {
                    std::this_thread::yield();
                }
            }

            total += local_sum;
        });
    }

    producer.join();
    for (auto& c : consumers) {
        c.join();
    }

    // Sum of 0..999 = 999 * 1000 / 2 = 499500
    EXPECT_EQ(total, 499500);
}
