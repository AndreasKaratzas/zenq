cmake_minimum_required(VERSION 3.20)
project(kernel_compute
    VERSION 0.1.0
    LANGUAGES CXX CUDA
    DESCRIPTION "High-performance kernel computation library"
)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# Export compilation database for tooling
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Set CUDA architectures for V100
set(CMAKE_CUDA_ARCHITECTURES 70)

# Compiler options
add_compile_options(
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>
)

# Add subdirectories
add_subdirectory(src)
add_subdirectory(tests)

# Installation
include(GNUInstallDirs)
install(
    DIRECTORY ${CMAKE_SOURCE_DIR}/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp" PATTERN "*.cuh"
)