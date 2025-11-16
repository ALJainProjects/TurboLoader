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
option(TURBOLOADER_WITH_CUDA "Build with CUDA support (GPU decode)" OFF)
option(TURBOLOADER_WITH_NCCL "Build with NCCL support (multi-GPU)" OFF)
option(TURBOLOADER_WITH_GLOO "Build with Gloo support (distributed)" OFF)

# Find dependencies
find_package(Threads REQUIRED)
find_package(JPEG REQUIRED)
find_package(PNG REQUIRED)
find_package(CURL REQUIRED)

# Prefer Python 3.13 over 3.14 (pybind11 compatibility)
set(Python3_FIND_STRATEGY VERSION)
set(Python3_FIND_VIRTUALENV FIRST)
find_program(PYTHON3_13_EXECUTABLE python3.13 PATHS /opt/homebrew/bin NO_DEFAULT_PATH)
if(PYTHON3_13_EXECUTABLE)
    set(Python3_EXECUTABLE ${PYTHON3_13_EXECUTABLE})
endif()

# Try to find WebP
find_library(WEBP_LIBRARY NAMES webp)
find_path(WEBP_INCLUDE_DIR NAMES webp/decode.h)

if(WEBP_LIBRARY AND WEBP_INCLUDE_DIR)
    set(WEBP_FOUND TRUE)
    message(STATUS "Found WebP: ${WEBP_LIBRARY}")
else()
    set(WEBP_FOUND FALSE)
    message(STATUS "WebP not found - WebP decoder will be stubbed")
endif()

# CUDA support (optional)
if(TURBOLOADER_WITH_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    message(STATUS "Building with CUDA support")

    # Find nvJPEG
    find_library(NVJPEG_LIBRARY NAMES nvjpeg HINTS ${CUDAToolkit_LIBRARY_DIR})
    if(NVJPEG_LIBRARY)
        message(STATUS "Found nvJPEG: ${NVJPEG_LIBRARY}")
    else()
        message(FATAL_ERROR "nvJPEG not found - required for GPU decode")
    endif()
endif()

# NCCL support (optional, requires CUDA)
if(TURBOLOADER_WITH_NCCL)
    if(NOT TURBOLOADER_WITH_CUDA)
        message(FATAL_ERROR "NCCL requires CUDA - enable TURBOLOADER_WITH_CUDA")
    endif()
    find_library(NCCL_LIBRARY NAMES nccl)
    find_path(NCCL_INCLUDE_DIR NAMES nccl.h)
    if(NCCL_LIBRARY AND NCCL_INCLUDE_DIR)
        message(STATUS "Found NCCL: ${NCCL_LIBRARY}")
    else()
        message(FATAL_ERROR "NCCL not found")
    endif()
endif()

# Gloo support (optional)
if(TURBOLOADER_WITH_GLOO)
    find_package(gloo REQUIRED)
    message(STATUS "Building with Gloo support")
endif()

# Main library sources
set(TURBOLOADER_SOURCES
    ${TURBOLOADER_SOURCE_DIR}/core/memory_pool.cpp
    ${TURBOLOADER_SOURCE_DIR}/core/lock_free_queue.cpp
    ${TURBOLOADER_SOURCE_DIR}/core/thread_pool.cpp
    ${TURBOLOADER_SOURCE_DIR}/readers/mmap_reader.cpp
    ${TURBOLOADER_SOURCE_DIR}/readers/tar_reader.cpp
    ${TURBOLOADER_SOURCE_DIR}/readers/storage_reader.cpp
    ${TURBOLOADER_SOURCE_DIR}/readers/http_reader.cpp
    ${TURBOLOADER_SOURCE_DIR}/readers/cached_reader.cpp
    ${TURBOLOADER_SOURCE_DIR}/decoders/jpeg_decoder.cpp
    ${TURBOLOADER_SOURCE_DIR}/decoders/image_decoder.cpp
    ${TURBOLOADER_SOURCE_DIR}/decoders/png_decoder.cpp
    ${TURBOLOADER_SOURCE_DIR}/decoders/webp_decoder.cpp
    ${TURBOLOADER_SOURCE_DIR}/transforms/image_transform.cpp
    ${TURBOLOADER_SOURCE_DIR}/transforms/simd_transforms.cpp
    ${TURBOLOADER_SOURCE_DIR}/transforms/augmentation_transforms.cpp
    ${TURBOLOADER_SOURCE_DIR}/pipeline/pipeline.cpp
)

# Add GPU sources if CUDA enabled
if(TURBOLOADER_WITH_CUDA)
    list(APPEND TURBOLOADER_SOURCES
        ${TURBOLOADER_SOURCE_DIR}/decoders/gpu_jpeg_decoder.cpp
    )
endif()

# Add distributed sources if enabled
if(TURBOLOADER_WITH_NCCL OR TURBOLOADER_WITH_GLOO)
    list(APPEND TURBOLOADER_SOURCES
        ${TURBOLOADER_SOURCE_DIR}/distributed/distributed_pipeline.cpp
    )
endif()

add_library(turboloader ${TURBOLOADER_SOURCES})

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
        JPEG::JPEG
        PNG::PNG
        CURL::libcurl
)

if(WEBP_FOUND)
    target_link_libraries(turboloader PUBLIC ${WEBP_LIBRARY})
    target_include_directories(turboloader PUBLIC ${WEBP_INCLUDE_DIR})
    target_compile_definitions(turboloader PUBLIC HAVE_WEBP)
endif()

# CUDA/nvJPEG support
if(TURBOLOADER_WITH_CUDA)
    target_link_libraries(turboloader PUBLIC
        CUDA::cudart
        ${NVJPEG_LIBRARY}
    )
    target_compile_definitions(turboloader PUBLIC TURBOLOADER_WITH_CUDA)
endif()

# NCCL support
if(TURBOLOADER_WITH_NCCL)
    target_link_libraries(turboloader PUBLIC ${NCCL_LIBRARY})
    target_include_directories(turboloader PUBLIC ${NCCL_INCLUDE_DIR})
    target_compile_definitions(turboloader PUBLIC TURBOLOADER_WITH_NCCL)
endif()

# Gloo support
if(TURBOLOADER_WITH_GLOO)
    target_link_libraries(turboloader PUBLIC gloo)
    target_compile_definitions(turboloader PUBLIC TURBOLOADER_WITH_GLOO)
endif()

# Enable CURL support
target_compile_definitions(turboloader PUBLIC HAVE_CURL)

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
