cmake_minimum_required(VERSION 3.20)
project(TurboLoader VERSION 1.0.0 LANGUAGES CXX)

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

# Find required dependencies
find_package(Threads REQUIRED)
find_package(JPEG REQUIRED)
find_package(PNG REQUIRED)
find_package(CURL REQUIRED)

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

# TurboLoader is header-only - create INTERFACE library
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
)

# WebP support
if(WEBP_FOUND)
    target_link_libraries(turboloader INTERFACE ${WEBP_LIBRARY})
    target_include_directories(turboloader INTERFACE ${WEBP_INCLUDE_DIR})
    target_compile_definitions(turboloader INTERFACE HAVE_WEBP)
endif()

# OpenCV support
if(TURBOLOADER_WITH_OPENCV)
    target_link_libraries(turboloader INTERFACE ${OpenCV_LIBS})
    target_include_directories(turboloader INTERFACE ${OpenCV_INCLUDE_DIRS})
    target_compile_definitions(turboloader INTERFACE HAVE_OPENCV)
endif()

# CUDA/nvJPEG support
if(TURBOLOADER_WITH_CUDA)
    target_link_libraries(turboloader INTERFACE
        CUDA::cudart
        ${NVJPEG_LIBRARY}
    )
    target_compile_definitions(turboloader INTERFACE TURBOLOADER_WITH_CUDA)
endif()

# Enable CURL support
target_compile_definitions(turboloader INTERFACE HAVE_CURL)

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
