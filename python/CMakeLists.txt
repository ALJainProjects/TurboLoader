# Python bindings using pybind11
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG QUIET)

if(NOT pybind11_FOUND)
    message(STATUS "pybind11 not found via find_package, trying pip location")
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(PYBIND11_CMAKE_DIR)
        list(APPEND CMAKE_PREFIX_PATH ${PYBIND11_CMAKE_DIR})
        find_package(pybind11 CONFIG REQUIRED)
    else()
        message(FATAL_ERROR "pybind11 not found. Install with: pip install pybind11")
    endif()
endif()

# Create Python module
pybind11_add_module(_turboloader
    turboloader_py.cpp
)

# Rename output to turboloader.so (without underscore)
set_target_properties(_turboloader PROPERTIES
    OUTPUT_NAME "turboloader"
)

target_link_libraries(_turboloader
    PRIVATE
        turboloader
)

target_include_directories(_turboloader
    PRIVATE
        ${TURBOLOADER_INCLUDE_DIR}
)

# Install Python module
install(TARGETS _turboloader
    LIBRARY DESTINATION ${Python3_SITEARCH}
)
