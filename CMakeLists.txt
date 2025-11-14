cmake_minimum_required(VERSION 3.20)
project(TurboLoader VERSION 0.1.0 LANGUAGES CXX)

# C++20 required for modern features
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags for performance
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -Wall -Wextra -Wpedantic")
    # Enable sanitizers in debug mode
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address,undefined")
    set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} -fsanitize=address,undefined")
endif()

# Project structure
set(TURBOLOADER_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(TURBOLOADER_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(TURBOLOADER_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tests)
set(TURBOLOADER_BENCHMARK_DIR ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks)

# Options
option(TURBOLOADER_BUILD_TESTS "Build tests" ON)
option(TURBOLOADER_BUILD_BENCHMARKS "Build benchmarks" ON)
option(TURBOLOADER_BUILD_PYTHON "Build Python bindings" ON)

# Find dependencies
find_package(Threads REQUIRED)

# Main library
add_library(turboloader
    ${TURBOLOADER_SOURCE_DIR}/core/memory_pool.cpp
    ${TURBOLOADER_SOURCE_DIR}/core/lock_free_queue.cpp
    ${TURBOLOADER_SOURCE_DIR}/core/thread_pool.cpp
    ${TURBOLOADER_SOURCE_DIR}/readers/mmap_reader.cpp
    ${TURBOLOADER_SOURCE_DIR}/readers/tar_reader.cpp
    ${TURBOLOADER_SOURCE_DIR}/pipeline/pipeline.cpp
)

target_include_directories(turboloader
    PUBLIC
        $<BUILD_INTERFACE:${TURBOLOADER_INCLUDE_DIR}>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${TURBOLOADER_SOURCE_DIR}
)

target_link_libraries(turboloader
    PUBLIC
        Threads::Threads
)

# Enable PIC for shared library
set_target_properties(turboloader PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

# Tests
if(TURBOLOADER_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Benchmarks
if(TURBOLOADER_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# Python bindings
if(TURBOLOADER_BUILD_PYTHON)
    add_subdirectory(python)
endif()

# Install
install(TARGETS turboloader
    EXPORT TurboLoaderTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

install(DIRECTORY ${TURBOLOADER_INCLUDE_DIR}/
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp"
)

# Export
install(EXPORT TurboLoaderTargets
    FILE TurboLoaderTargets.cmake
    NAMESPACE TurboLoader::
    DESTINATION lib/cmake/TurboLoader
)
