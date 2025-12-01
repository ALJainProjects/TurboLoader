/**
 * @file compat.hpp
 * @brief C++17/C++20 compatibility utilities
 *
 * Provides std::span polyfill for C++17 compilers.
 * When compiled with C++20, uses the standard library implementation.
 */

#pragma once

#if __cplusplus >= 202002L && __has_include(<span>)
    // C++20: use standard span
    #include <span>
    namespace turboloader {
        using std::span;
    }
#else
    // C++17: minimal span polyfill
    #include <cstddef>
    #include <type_traits>
    #include <array>
    #include <vector>

    namespace turboloader {

    /**
     * @brief Minimal std::span polyfill for C++17
     *
     * Only implements the subset of std::span used by TurboLoader:
     * - Construction from pointer + size
     * - Construction from contiguous containers
     * - data(), size(), empty()
     * - operator[]
     * - begin(), end()
     */
    template<typename T>
    class span {
    public:
        using element_type = T;
        using value_type = std::remove_cv_t<T>;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using pointer = T*;
        using const_pointer = const T*;
        using reference = T&;
        using const_reference = const T&;
        using iterator = pointer;
        using const_iterator = const_pointer;

        // Default constructor - empty span
        constexpr span() noexcept : data_(nullptr), size_(0) {}

        // Constructor from pointer and size
        constexpr span(pointer ptr, size_type count) noexcept
            : data_(ptr), size_(count) {}

        // Constructor from pointer pair
        constexpr span(pointer first, pointer last) noexcept
            : data_(first), size_(static_cast<size_type>(last - first)) {}

        // Constructor from C-style array
        template<std::size_t N>
        constexpr span(T (&arr)[N]) noexcept
            : data_(arr), size_(N) {}

        // Constructor from std::array
        template<std::size_t N>
        constexpr span(std::array<value_type, N>& arr) noexcept
            : data_(arr.data()), size_(N) {}

        template<std::size_t N>
        constexpr span(const std::array<value_type, N>& arr) noexcept
            : data_(arr.data()), size_(N) {}

        // Constructor from std::vector
        template<typename Allocator>
        constexpr span(std::vector<value_type, Allocator>& vec) noexcept
            : data_(vec.data()), size_(vec.size()) {}

        template<typename Allocator>
        constexpr span(const std::vector<value_type, Allocator>& vec) noexcept
            : data_(vec.data()), size_(vec.size()) {}

        // Copy constructor (span is a view, copying is fine)
        constexpr span(const span& other) noexcept = default;

        // Assignment
        constexpr span& operator=(const span& other) noexcept = default;

        // Accessors
        constexpr pointer data() const noexcept { return data_; }
        constexpr size_type size() const noexcept { return size_; }
        constexpr size_type size_bytes() const noexcept { return size_ * sizeof(T); }
        constexpr bool empty() const noexcept { return size_ == 0; }

        // Element access
        constexpr reference operator[](size_type idx) const noexcept {
            return data_[idx];
        }

        constexpr reference front() const noexcept { return data_[0]; }
        constexpr reference back() const noexcept { return data_[size_ - 1]; }

        // Iterators
        constexpr iterator begin() const noexcept { return data_; }
        constexpr iterator end() const noexcept { return data_ + size_; }

        // Subviews
        constexpr span first(size_type count) const noexcept {
            return span(data_, count);
        }

        constexpr span last(size_type count) const noexcept {
            return span(data_ + (size_ - count), count);
        }

        constexpr span subspan(size_type offset, size_type count = static_cast<size_type>(-1)) const noexcept {
            if (count == static_cast<size_type>(-1)) {
                return span(data_ + offset, size_ - offset);
            }
            return span(data_ + offset, count);
        }

    private:
        pointer data_;
        size_type size_;
    };

    // Deduction guides
    template<typename T, std::size_t N>
    span(T (&)[N]) -> span<T>;

    template<typename T, std::size_t N>
    span(std::array<T, N>&) -> span<T>;

    template<typename T, std::size_t N>
    span(const std::array<T, N>&) -> span<const T>;

    template<typename T, typename Allocator>
    span(std::vector<T, Allocator>&) -> span<T>;

    template<typename T, typename Allocator>
    span(const std::vector<T, Allocator>&) -> span<const T>;

    } // namespace turboloader

    // Also inject into std namespace for compatibility with existing code
    namespace std {
        using turboloader::span;
    }
#endif
