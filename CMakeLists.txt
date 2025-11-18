cmake_minimum_required(VERSION 3.20)
project(TurboLoader VERSION 1.5.1 LANGUAGES CXX)

# C++20 required for modern features (std::span, concepts, etc.)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -Wall -Wextra -Wpedantic")
endif()

# Options
option(TURBOLOADER_BUILD_TESTS "Build tests" ON)
option(TURBOLOADER_BUILD_PYTHON "Build Python bindings" ON)
option(TURBOLOADER_WITH_CUDA "Build with CUDA support (GPU decode)" OFF)
option(TURBOLOADER_WITH_OPENCV "Build with OpenCV (video support)" OFF)
option(ENABLE_CUDA "Enable CUDA support for multi-GPU pipeline" OFF)
option(ENABLE_MPI "Enable MPI support for distributed training" OFF)

# Find required dependencies
find_package(Threads REQUIRED)
find_package(JPEG REQUIRED)
find_package(PNG REQUIRED)
find_package(CURL REQUIRED)

# Find LZ4 for TBL v2 compression
find_library(LZ4_LIBRARY NAMES lz4)
find_path(LZ4_INCLUDE_DIR NAMES lz4.h)

if(LZ4_LIBRARY AND LZ4_INCLUDE_DIR)
    set(LZ4_FOUND TRUE)
    message(STATUS "Found LZ4: ${LZ4_LIBRARY}")
else()
    message(FATAL_ERROR "LZ4 not found - required for TBL v2 format")
endif()

# Try to find WebP (optional)
find_library(WEBP_LIBRARY NAMES webp)
find_path(WEBP_INCLUDE_DIR NAMES webp/decode.h)

if(WEBP_LIBRARY AND WEBP_INCLUDE_DIR)
    set(WEBP_FOUND TRUE)
    message(STATUS "Found WebP: ${WEBP_LIBRARY}")
else()
    set(WEBP_FOUND FALSE)
    message(STATUS "WebP not found - WebP decoder will be disabled")
endif()

# OpenCV for video support (optional)
if(TURBOLOADER_WITH_OPENCV)
    find_package(OpenCV REQUIRED)
    message(STATUS "Found OpenCV: ${OpenCV_VERSION}")
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

# MPI support for distributed training (optional)
if(ENABLE_MPI)
    find_package(MPI REQUIRED)
    message(STATUS "Building with MPI support")
endif()

# Additional CUDA for multi-GPU pipeline (optional)
if(ENABLE_CUDA)
    if(NOT TURBOLOADER_WITH_CUDA)
        enable_language(CUDA)
        find_package(CUDAToolkit REQUIRED)
    endif()
    message(STATUS "Building with CUDA multi-GPU support")
endif()

# Collect source files for non-header-only components
set(TURBOLOADER_SOURCES "")

# Add multi-GPU pipeline if CUDA enabled
if(ENABLE_CUDA)
    list(APPEND TURBOLOADER_SOURCES
        src/gpu/multi_gpu_pipeline.cpp
    )
endif()

# Add distributed pipeline if MPI enabled
if(ENABLE_MPI)
    list(APPEND TURBOLOADER_SOURCES
        src/distributed/distributed_pipeline.cpp
    )
endif()

# Create library (header-only if no sources, otherwise normal library)
if(TURBOLOADER_SOURCES)
    add_library(turboloader ${TURBOLOADER_SOURCES})

    target_include_directories(turboloader
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
            $<INSTALL_INTERFACE:include>
    )

    target_link_libraries(turboloader
        PUBLIC
            Threads::Threads
            JPEG::JPEG
            PNG::PNG
            CURL::libcurl
            ${LZ4_LIBRARY}
    )

    target_include_directories(turboloader
        PUBLIC
            ${LZ4_INCLUDE_DIR}
    )
else()
    # Header-only fallback
    add_library(turboloader INTERFACE)

    target_include_directories(turboloader
        INTERFACE
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
            $<INSTALL_INTERFACE:include>
    )

    target_link_libraries(turboloader
        INTERFACE
            Threads::Threads
            JPEG::JPEG
            PNG::PNG
            CURL::libcurl
            ${LZ4_LIBRARY}
    )

    target_include_directories(turboloader
        INTERFACE
            ${LZ4_INCLUDE_DIR}
    )
endif()

# Determine link scope based on library type
if(TURBOLOADER_SOURCES)
    set(LINK_SCOPE PUBLIC)
else()
    set(LINK_SCOPE INTERFACE)
endif()

# WebP support
if(WEBP_FOUND)
    target_link_libraries(turboloader ${LINK_SCOPE} ${WEBP_LIBRARY})
    target_include_directories(turboloader ${LINK_SCOPE} ${WEBP_INCLUDE_DIR})
    target_compile_definitions(turboloader ${LINK_SCOPE} HAVE_WEBP)
endif()

# OpenCV support
if(TURBOLOADER_WITH_OPENCV)
    target_link_libraries(turboloader ${LINK_SCOPE} ${OpenCV_LIBS})
    target_include_directories(turboloader ${LINK_SCOPE} ${OpenCV_INCLUDE_DIRS})
    target_compile_definitions(turboloader ${LINK_SCOPE} HAVE_OPENCV)
endif()

# CUDA/nvJPEG support
if(TURBOLOADER_WITH_CUDA)
    target_link_libraries(turboloader ${LINK_SCOPE}
        CUDA::cudart
        ${NVJPEG_LIBRARY}
    )
    target_compile_definitions(turboloader ${LINK_SCOPE} TURBOLOADER_WITH_CUDA)
endif()

# Multi-GPU CUDA support
if(ENABLE_CUDA)
    target_link_libraries(turboloader ${LINK_SCOPE} CUDA::cudart)
    target_compile_definitions(turboloader ${LINK_SCOPE} TURBOLOADER_ENABLE_CUDA)
endif()

# MPI support
if(ENABLE_MPI)
    target_link_libraries(turboloader ${LINK_SCOPE} MPI::MPI_CXX)
    target_include_directories(turboloader ${LINK_SCOPE} ${MPI_CXX_INCLUDE_DIRS})
    target_compile_definitions(turboloader ${LINK_SCOPE} TURBOLOADER_ENABLE_MPI)
endif()

# Enable CURL support
target_compile_definitions(turboloader ${LINK_SCOPE} HAVE_CURL)

# Command-line tools (v1.5.0)
# TAR to TBL v2 converter tool (with LZ4 compression)
add_executable(tar_to_tbl tools/tar_to_tbl_v2.cpp)
target_include_directories(tar_to_tbl PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${LZ4_INCLUDE_DIR})
target_link_libraries(tar_to_tbl PRIVATE turboloader ${LZ4_LIBRARY})
install(TARGETS tar_to_tbl DESTINATION bin)

# Tests
if(TURBOLOADER_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Python bindings (handled via setup.py, not CMake)
# if(TURBOLOADER_BUILD_PYTHON)
#     add_subdirectory(src/python)
# endif()

# Install header files
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/
    DESTINATION include/turboloader
    FILES_MATCHING PATTERN "*.hpp"
)

# Install targets
install(TARGETS turboloader
    EXPORT TurboLoaderTargets
    INCLUDES DESTINATION include
)

# Export
install(EXPORT TurboLoaderTargets
    FILE TurboLoaderTargets.cmake
    NAMESPACE TurboLoader::
    DESTINATION lib/cmake/TurboLoader
)
