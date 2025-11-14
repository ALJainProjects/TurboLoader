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
