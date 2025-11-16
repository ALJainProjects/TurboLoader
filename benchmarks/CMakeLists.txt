# Basic benchmark (recommended for users)
add_executable(basic_benchmark
    basic_benchmark.cpp
)

target_link_libraries(basic_benchmark
    PRIVATE
        turboloader
)

target_include_directories(basic_benchmark
    PRIVATE
        ${TURBOLOADER_INCLUDE_DIR}
)

# Multi-format benchmark
add_executable(multiformat_benchmark
    multiformat_benchmark.cpp
)

target_link_libraries(multiformat_benchmark
    PRIVATE
        turboloader
)

target_include_directories(multiformat_benchmark
    PRIVATE
        ${TURBOLOADER_INCLUDE_DIR}
)

# SIMD transform benchmark
add_executable(benchmark_simd
    benchmark_simd.cpp
)

target_link_libraries(benchmark_simd
    PRIVATE
        turboloader
)

target_include_directories(benchmark_simd
    PRIVATE
        ${TURBOLOADER_INCLUDE_DIR}
)
