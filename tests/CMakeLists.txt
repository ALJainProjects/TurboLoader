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
if(ENABLE_CUDA)
    add_executable(test_multi_gpu test_multi_gpu.cpp)
    target_include_directories(test_multi_gpu PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(test_multi_gpu PRIVATE turboloader CUDA::cudart)
    target_compile_definitions(test_multi_gpu PRIVATE TURBOLOADER_ENABLE_CUDA)
    add_test(NAME multi_gpu COMMAND test_multi_gpu)
    message(STATUS "Building multi-GPU test with CUDA support")
else()
    # Build test without CUDA (will test error handling)
    add_executable(test_multi_gpu test_multi_gpu.cpp)
    target_include_directories(test_multi_gpu PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(test_multi_gpu PRIVATE JPEG::JPEG PNG::PNG CURL::libcurl)
    add_test(NAME multi_gpu COMMAND test_multi_gpu)
    message(STATUS "Building multi-GPU test without CUDA (error handling only)")
endif()

# Test executable for distributed pipeline (v0.5.2)
if(ENABLE_MPI)
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
    # Build test without MPI (will test error handling)
    add_executable(test_distributed test_distributed.cpp)
    target_include_directories(test_distributed PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(test_distributed PRIVATE JPEG::JPEG PNG::PNG CURL::libcurl)
    add_test(NAME distributed COMMAND test_distributed)
    message(STATUS "Building distributed test without MPI (error handling only)")
endif()

# Additional tests will be added as we implement more components
# add_executable(test_core test_core.cpp)
