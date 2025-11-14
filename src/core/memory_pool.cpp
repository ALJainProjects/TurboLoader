#include "turboloader/core/memory_pool.hpp"
#include <algorithm>
#include <stdexcept>

namespace turboloader {

MemoryPool::MemoryPool(size_t block_size, size_t alignment)
    : block_size_(block_size)
    , default_alignment_(alignment)
    , current_block_(0)
    , total_bytes_(0)
    , used_bytes_(0) {

    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        throw std::invalid_argument("Alignment must be power of 2");
    }

    // Pre-allocate first block
    allocate_new_block(block_size_);
}

MemoryPool::~MemoryPool() {
    clear();
}

void* MemoryPool::allocate(size_t size, size_t alignment) {
    if (size == 0) {
        return nullptr;
    }

    if (alignment == 0) {
        alignment = default_alignment_;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    // Try current block first
    if (current_block_ < blocks_.size()) {
        Block& block = blocks_[current_block_];

        // Align current position
        size_t aligned_pos = align_up(block.used, alignment);
        size_t end_pos = aligned_pos + size;

        if (end_pos <= block.size) {
            // Fits in current block
            void* ptr = block.data.get() + aligned_pos;
            used_bytes_ += (end_pos - block.used);
            block.used = end_pos;
            return ptr;
        }
    }

    // Need new block
    size_t required_size = std::max(size + alignment, block_size_);
    allocate_new_block(required_size);

    // Retry allocation in new block
    Block& block = blocks_[current_block_];
    size_t aligned_pos = align_up(block.used, alignment);
    size_t end_pos = aligned_pos + size;

    if (end_pos > block.size) {
        // Should never happen unless size > required_size
        throw std::bad_alloc();
    }

    void* ptr = block.data.get() + aligned_pos;
    used_bytes_ += (end_pos - block.used);
    block.used = end_pos;
    return ptr;
}

void MemoryPool::reset() {
    std::lock_guard<std::mutex> lock(mutex_);

    // Reset all blocks to reuse memory
    for (auto& block : blocks_) {
        block.used = 0;
    }

    current_block_ = 0;
    used_bytes_ = 0;
}

void MemoryPool::clear() {
    std::lock_guard<std::mutex> lock(mutex_);

    blocks_.clear();
    current_block_ = 0;
    total_bytes_ = 0;
    used_bytes_ = 0;
}

void MemoryPool::allocate_new_block(size_t min_size) {
    // Already holding mutex from allocate()

    size_t new_size = std::max(min_size, block_size_);

    blocks_.emplace_back(new_size);
    current_block_ = blocks_.size() - 1;
    total_bytes_ += new_size;
}

}  // namespace turboloader
