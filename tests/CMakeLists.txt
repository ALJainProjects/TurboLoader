# Test suite for TurboLoader

# Test executable for TAR reader
add_executable(test_tar_reader test_tar_reader.cpp)
target_link_libraries(test_tar_reader PRIVATE turboloader)
add_test(NAME tar_reader COMMAND test_tar_reader)

# Test executable for image decoder (JPEG, PNG, WebP, BMP, TIFF)
add_executable(test_image_decoder test_image_decoder.cpp)
target_link_libraries(test_image_decoder PRIVATE turboloader)
add_test(NAME image_decoder COMMAND test_image_decoder)

# Test executable for HTTP reader with connection pooling
find_package(CURL REQUIRED)
add_executable(test_http_reader test_http_reader.cpp)
target_link_libraries(test_http_reader PRIVATE CURL::libcurl)
target_include_directories(test_http_reader PRIVATE ${CMAKE_SOURCE_DIR}/src)
add_test(NAME http_reader COMMAND test_http_reader)

# Test executable for S3 reader (uses HTTP fallback without AWS SDK)
add_executable(test_s3_reader test_s3_reader.cpp)
target_link_libraries(test_s3_reader PRIVATE CURL::libcurl)
target_include_directories(test_s3_reader PRIVATE ${CMAKE_SOURCE_DIR}/src)
add_test(NAME s3_reader COMMAND test_s3_reader)

# Test executable for GCS reader (uses HTTP fallback without Google Cloud SDK)
add_executable(test_gcs_reader test_gcs_reader.cpp)
target_link_libraries(test_gcs_reader PRIVATE CURL::libcurl)
target_include_directories(test_gcs_reader PRIVATE ${CMAKE_SOURCE_DIR}/src)
add_test(NAME gcs_reader COMMAND test_gcs_reader)

# Test executable for unified reader orchestrator (auto-selects reader based on source)
add_executable(test_reader_orchestrator test_reader_orchestrator.cpp)
target_link_libraries(test_reader_orchestrator PRIVATE CURL::libcurl)
target_include_directories(test_reader_orchestrator PRIVATE ${CMAKE_SOURCE_DIR}/src)
add_test(NAME reader_orchestrator COMMAND test_reader_orchestrator)

# Test executable for video decoder (requires FFmpeg)
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(FFMPEG QUIET
        libavcodec
        libavformat
        libavutil
        libswscale
    )
endif()

if(FFMPEG_FOUND)
    add_executable(test_video_decoder test_video_decoder.cpp)
    target_include_directories(test_video_decoder PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${FFMPEG_INCLUDE_DIRS}
    )
    target_link_directories(test_video_decoder PRIVATE ${FFMPEG_LIBRARY_DIRS})
    target_link_libraries(test_video_decoder PRIVATE ${FFMPEG_LIBRARIES})
    target_compile_definitions(test_video_decoder PRIVATE HAVE_FFMPEG)
    add_test(NAME video_decoder COMMAND test_video_decoder)
    message(STATUS "Building video decoder test with FFmpeg support")
else()
    # Build test without FFmpeg (will show skipped tests)
    add_executable(test_video_decoder test_video_decoder.cpp)
    target_include_directories(test_video_decoder PRIVATE ${CMAKE_SOURCE_DIR}/src)
    add_test(NAME video_decoder COMMAND test_video_decoder)
    message(STATUS "Building video decoder test without FFmpeg (tests will be skipped)")
endif()

# Test executable for CSV decoder (header-only, no dependencies)
add_executable(test_csv_decoder test_csv_decoder.cpp)
target_include_directories(test_csv_decoder PRIVATE ${CMAKE_SOURCE_DIR}/src)
add_test(NAME csv_decoder COMMAND test_csv_decoder)

# Test executable for Parquet decoder (requires Apache Arrow)
if(PkgConfig_FOUND)
    pkg_check_modules(ARROW QUIET arrow parquet)
endif()

if(ARROW_FOUND)
    add_executable(test_parquet_decoder test_parquet_decoder.cpp)
    target_include_directories(test_parquet_decoder PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${ARROW_INCLUDE_DIRS}
    )
    target_link_directories(test_parquet_decoder PRIVATE ${ARROW_LIBRARY_DIRS})
    target_link_libraries(test_parquet_decoder PRIVATE ${ARROW_LIBRARIES})
    target_compile_definitions(test_parquet_decoder PRIVATE HAVE_ARROW)
    add_test(NAME parquet_decoder COMMAND test_parquet_decoder)
    message(STATUS "Building Parquet decoder test with Apache Arrow support")
else()
    # Build test without Arrow (will show skipped tests)
    add_executable(test_parquet_decoder test_parquet_decoder.cpp)
    target_include_directories(test_parquet_decoder PRIVATE ${CMAKE_SOURCE_DIR}/src)
    add_test(NAME parquet_decoder COMMAND test_parquet_decoder)
    message(STATUS "Building Parquet decoder test without Apache Arrow (tests will be skipped)")
endif()

# Test executable for unified pipeline (all formats, all sources)
add_executable(test_unified_pipeline test_unified_pipeline.cpp)
target_include_directories(test_unified_pipeline PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_unified_pipeline PRIVATE turboloader)
add_test(NAME unified_pipeline COMMAND test_unified_pipeline)

# Test executable for nvJPEG decoder (GPU-accelerated JPEG with CPU fallback)
# Use modern CMake CUDA language support instead of deprecated FindCUDA
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_library(NVJPEG_LIBRARY nvjpeg HINTS ${CMAKE_CUDA_TOOLKIT_ROOT_DIR}/lib64 ${CMAKE_CUDA_TOOLKIT_ROOT_DIR}/lib)
    if(NVJPEG_LIBRARY)
        add_executable(test_nvjpeg_decoder test_nvjpeg_decoder.cpp)
        target_include_directories(test_nvjpeg_decoder PRIVATE
            ${CMAKE_SOURCE_DIR}/src
            ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
        )
        target_link_libraries(test_nvjpeg_decoder PRIVATE
            turboloader
            ${NVJPEG_LIBRARY}
            CUDA::cudart
        )
        target_compile_definitions(test_nvjpeg_decoder PRIVATE HAVE_NVJPEG)
        add_test(NAME nvjpeg_decoder COMMAND test_nvjpeg_decoder)
        message(STATUS "Building nvJPEG decoder test with GPU support")
    else()
        # Build test without nvJPEG (will use CPU fallback)
        add_executable(test_nvjpeg_decoder test_nvjpeg_decoder.cpp)
        target_include_directories(test_nvjpeg_decoder PRIVATE ${CMAKE_SOURCE_DIR}/src)
        target_link_libraries(test_nvjpeg_decoder PRIVATE turboloader)
        add_test(NAME nvjpeg_decoder COMMAND test_nvjpeg_decoder)
        message(STATUS "Building nvJPEG decoder test without GPU (CPU fallback only)")
    endif()
else()
    # Build test without CUDA (CPU fallback only)
    add_executable(test_nvjpeg_decoder test_nvjpeg_decoder.cpp)
    target_include_directories(test_nvjpeg_decoder PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(test_nvjpeg_decoder PRIVATE turboloader)
    add_test(NAME nvjpeg_decoder COMMAND test_nvjpeg_decoder)
    message(STATUS "Building nvJPEG decoder test without CUDA (CPU fallback only)")
endif()

# Test executable for multi-GPU pipeline (v0.5.2)
# Only build and test on Linux with CUDA support
if(ENABLE_CUDA AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_executable(test_multi_gpu test_multi_gpu.cpp)
    target_include_directories(test_multi_gpu PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(test_multi_gpu PRIVATE turboloader CUDA::cudart)
    target_compile_definitions(test_multi_gpu PRIVATE TURBOLOADER_ENABLE_CUDA)
    add_test(NAME multi_gpu COMMAND test_multi_gpu)
    message(STATUS "Building multi-GPU test with CUDA support")
else()
    message(STATUS "Skipping multi-GPU test (requires CUDA on Linux)")
endif()

# Test executable for distributed pipeline (v0.5.2)
# Only build and test on Linux with MPI support
if(ENABLE_MPI AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_executable(test_distributed test_distributed.cpp)
    target_include_directories(test_distributed PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${MPI_CXX_INCLUDE_DIRS}
    )
    target_link_libraries(test_distributed PRIVATE turboloader MPI::MPI_CXX)
    target_compile_definitions(test_distributed PRIVATE TURBOLOADER_ENABLE_MPI)
    add_test(NAME distributed COMMAND test_distributed)
    message(STATUS "Building distributed test with MPI support")
else()
    message(STATUS "Skipping distributed test (requires MPI on Linux)")
endif()

# Test executable for transforms (SIMD-accelerated, v0.6.0)
# Find Google Test
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG release-1.12.1
)
FetchContent_MakeAvailable(googletest)

add_executable(test_transforms test_transforms.cpp)
target_include_directories(test_transforms PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_transforms PRIVATE turboloader gtest gtest_main)
add_test(NAME transforms COMMAND test_transforms)
message(STATUS "Building transform tests with SIMD support")

# Test executable for advanced transforms (v0.7.0)
add_executable(test_advanced_transforms test_advanced_transforms.cpp)
target_include_directories(test_advanced_transforms PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_advanced_transforms PRIVATE turboloader gtest gtest_main)
add_test(NAME advanced_transforms COMMAND test_advanced_transforms)
message(STATUS "Building advanced transform tests (v0.7.0)")

# Test executable for AVX-512 SIMD optimizations (v1.1.0)
add_executable(test_avx512_simd test_avx512_simd.cpp)
target_include_directories(test_avx512_simd PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_avx512_simd PRIVATE turboloader)
add_test(NAME avx512_simd COMMAND test_avx512_simd)
message(STATUS "Building AVX-512 SIMD tests (v1.1.0)")

# Test executable for SIMD HWC→CHW transpose (v2.12.0 Phase 3.1)
add_executable(test_simd_transpose test_simd_transpose.cpp)
target_include_directories(test_simd_transpose PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_simd_transpose PRIVATE gtest gtest_main)
add_test(NAME simd_transpose COMMAND test_simd_transpose)
message(STATUS "Building SIMD HWC→CHW transpose tests (v2.12.0)")

# Test executable for SIMD bilinear interpolation (v2.13.0 Phase 3.2)
add_executable(test_simd_bilinear test_simd_bilinear.cpp)
target_include_directories(test_simd_bilinear PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_simd_bilinear PRIVATE turboloader gtest gtest_main)
add_test(NAME simd_bilinear COMMAND test_simd_bilinear)
message(STATUS "Building SIMD bilinear interpolation tests (v2.13.0)")

# Test executable for HybridWaitStrategy (v2.14.0 Phase 4.1)
add_executable(test_hybrid_wait_strategy test_hybrid_wait_strategy.cpp)
target_include_directories(test_hybrid_wait_strategy PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_hybrid_wait_strategy PRIVATE gtest gtest_main pthread)
add_test(NAME hybrid_wait_strategy COMMAND test_hybrid_wait_strategy)
message(STATUS "Building HybridWaitStrategy tests (v2.14.0)")

# Test executable for prefetch pipeline (v1.1.0)
add_executable(test_prefetch_pipeline test_prefetch_pipeline.cpp)
target_include_directories(test_prefetch_pipeline PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_prefetch_pipeline PRIVATE turboloader)
add_test(NAME prefetch_pipeline COMMAND test_prefetch_pipeline)
message(STATUS "Building prefetch pipeline tests (v1.1.0)")

# Test executable for TBL v2 format (v1.5.0)
add_executable(test_tbl_v2 test_tbl_v2.cpp)
target_include_directories(test_tbl_v2 PRIVATE ${CMAKE_SOURCE_DIR}/src ${LZ4_INCLUDE_DIR})
target_link_libraries(test_tbl_v2 PRIVATE turboloader ${LZ4_LIBRARY})
add_test(NAME tbl_v2 COMMAND test_tbl_v2)
message(STATUS "Building TBL v2 format tests with LZ4 compression (v1.5.0)")

# Test executable for Smart Batching (v1.2.0)
add_executable(test_smart_batching test_smart_batching.cpp)
target_include_directories(test_smart_batching PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_smart_batching PRIVATE turboloader)
add_test(NAME smart_batching COMMAND test_smart_batching)
message(STATUS "Building Smart Batching tests (v1.2.0)")

# Test executable for Pipeline GPU Decode Integration (v1.2.1)
# Tests nvJPEG integration into UnifiedPipeline
# GPU tests will run CPU fallback if nvJPEG unavailable
add_executable(test_pipeline_gpu_decode test_pipeline_gpu_decode.cpp)
target_include_directories(test_pipeline_gpu_decode PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_pipeline_gpu_decode PRIVATE turboloader)
# Add nvJPEG support if available
if(CMAKE_CUDA_COMPILER AND NVJPEG_LIBRARY)
    target_compile_definitions(test_pipeline_gpu_decode PRIVATE HAVE_NVJPEG)
    target_link_libraries(test_pipeline_gpu_decode PRIVATE
        ${NVJPEG_LIBRARY}
        CUDA::cudart
    )
    message(STATUS "Building Pipeline GPU Decode tests with nvJPEG support (v1.2.1)")
else()
    message(STATUS "Building Pipeline GPU Decode tests with CPU fallback only (v1.2.1)")
endif()
add_test(NAME pipeline_gpu_decode COMMAND test_pipeline_gpu_decode)

# Test executable for v2.5.0 batch array transfer functionality
add_executable(test_batch_array test_batch_array.cpp)
target_include_directories(test_batch_array PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_batch_array PRIVATE turboloader)
add_test(NAME batch_array COMMAND test_batch_array)
message(STATUS "Building batch array transfer tests (v2.5.0)")

# Test executable for Phase 2: DCT-scaled JPEG decoding
add_executable(test_decode_scaled test_decode_scaled.cpp)
target_include_directories(test_decode_scaled PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_decode_scaled PRIVATE turboloader)
add_test(NAME decode_scaled COMMAND test_decode_scaled)
message(STATUS "Building DCT-scaled decode tests (v2.5.0 Phase 2)")

# Test executable for Phase 3: Hybrid wait strategy
add_executable(test_hybrid_wait test_hybrid_wait.cpp)
target_include_directories(test_hybrid_wait PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_hybrid_wait PRIVATE turboloader pthread)
add_test(NAME hybrid_wait COMMAND test_hybrid_wait)
message(STATUS "Building hybrid wait tests (v2.5.0 Phase 3)")

# Test executable for AutoAugment operations (v2.8.0)
add_executable(test_autoaugment_ops test_autoaugment_ops.cpp)
target_include_directories(test_autoaugment_ops PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_autoaugment_ops PRIVATE turboloader gtest gtest_main)
add_test(NAME autoaugment_ops COMMAND test_autoaugment_ops)
message(STATUS "Building AutoAugment operations tests (v2.8.0)")

# Test executable for BufferPool (v2.10.0)
add_executable(test_buffer_pool test_buffer_pool.cpp)
target_include_directories(test_buffer_pool PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_buffer_pool PRIVATE turboloader gtest gtest_main)
add_test(NAME buffer_pool COMMAND test_buffer_pool)
message(STATUS "Building BufferPool tests (v2.10.0)")

# Additional tests will be added as we implement more components
# add_executable(test_core test_core.cpp)
