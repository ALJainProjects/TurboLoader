# Download and setup Google Test
include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Enable testing
enable_testing()
include(GoogleTest)

# Test executable
add_executable(turboloader_tests
    test_lock_free_queue.cpp
    test_memory_pool.cpp
    test_mmap_reader.cpp
)

target_link_libraries(turboloader_tests
    PRIVATE
        turboloader
        GTest::gtest_main
)

target_include_directories(turboloader_tests
    PRIVATE
        ${TURBOLOADER_INCLUDE_DIR}
)

# Discover tests
gtest_discover_tests(turboloader_tests)

# HTTP Reader test (standalone, requires network)
add_executable(test_http_reader test_http_reader.cpp)
target_link_libraries(test_http_reader PRIVATE turboloader)
target_include_directories(test_http_reader PRIVATE ${TURBOLOADER_INCLUDE_DIR})

# Transforms test
add_executable(test_transforms test_transforms.cpp)
target_link_libraries(test_transforms PRIVATE turboloader)
target_include_directories(test_transforms PRIVATE ${TURBOLOADER_INCLUDE_DIR})

# SIMD Transforms test
add_executable(test_simd_transforms test_simd_transforms.cpp)
target_link_libraries(test_simd_transforms PRIVATE turboloader)
target_include_directories(test_simd_transforms PRIVATE ${TURBOLOADER_INCLUDE_DIR})
