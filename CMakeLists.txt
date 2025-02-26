cmake_minimum_required(VERSION 3.20)
project(hpc_tensor VERSION 1.0.0 LANGUAGES CUDA CXX)

# ===================================================================
# Project configuration and options
# ===================================================================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Add this line to include the module
include(CheckCXXCompilerFlag)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Build options
option(HPC_ENABLE_CUDA "Enable CUDA support" ON)
option(HPC_ENABLE_LOGGING "Enable performance and optimization logging" ON)
option(HPC_COLORED_LOGGING "Enable colored console output in logs" ON)
option(HPC_BUILD_TESTS "Build tests" ON)
option(HPC_BUILD_BENCHMARKS "Build benchmarks" OFF)

# ===================================================================
# Benchmarks
# ===================================================================
if(HPC_BUILD_BENCHMARKS)
    # Find Google benchmark
    find_package(benchmark REQUIRED)
    
    # Add benchmarks subdirectory
    add_subdirectory(benchmark)
    
    message(STATUS "Benchmarks enabled")
else()
    message(STATUS "Benchmarks disabled")
endif()

# Define installation directories
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/hpc_tensor)

# Set default build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type: Debug, Release, RelWithDebInfo, MinSizeRel" FORCE)
endif()

# ===================================================================
# Compiler flags and optimizations
# ===================================================================

# Base compiler flags
set(BASE_CXX_FLAGS "")
set(BASE_CXX_FLAGS_DEBUG "")
set(BASE_CXX_FLAGS_RELEASE "")

# Check and add CPU architecture specific optimizations
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(BASE_CXX_FLAGS "${BASE_CXX_FLAGS} -Wall -Wextra")
    set(BASE_CXX_FLAGS_DEBUG "${BASE_CXX_FLAGS_DEBUG} -g -O0")
    set(BASE_CXX_FLAGS_RELEASE "${BASE_CXX_FLAGS_RELEASE} -O3 -ffast-math")
    
    if(NOT APPLE)
        set(BASE_CXX_FLAGS_RELEASE "${BASE_CXX_FLAGS_RELEASE} -march=native")
    endif()
endif()

# Add SIMD instruction set support checks
check_cxx_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512)
check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
check_cxx_compiler_flag("-mavx" COMPILER_SUPPORTS_AVX)
check_cxx_compiler_flag("-mfma" COMPILER_SUPPORTS_FMA)
check_cxx_compiler_flag("-msse4.2" COMPILER_SUPPORTS_SSE42)

# Enable supported SIMD instruction sets
if(COMPILER_SUPPORTS_AVX512)
    set(BASE_CXX_FLAGS_RELEASE "${BASE_CXX_FLAGS_RELEASE} -mavx512f")
    add_definitions(-DHAVE_AVX512)
endif()

if(COMPILER_SUPPORTS_AVX2)
    set(BASE_CXX_FLAGS_RELEASE "${BASE_CXX_FLAGS_RELEASE} -mavx2")
    add_definitions(-DHAVE_AVX2)
endif()

if(COMPILER_SUPPORTS_FMA)
    set(BASE_CXX_FLAGS_RELEASE "${BASE_CXX_FLAGS_RELEASE} -mfma")
    add_definitions(-DHAVE_FMA)
endif()

if(COMPILER_SUPPORTS_AVX)
    set(BASE_CXX_FLAGS_RELEASE "${BASE_CXX_FLAGS_RELEASE} -mavx")
    add_definitions(-DHAVE_AVX)
endif()

if(COMPILER_SUPPORTS_SSE42)
    set(BASE_CXX_FLAGS_RELEASE "${BASE_CXX_FLAGS_RELEASE} -msse4.2")
    add_definitions(-DHAVE_SSE42)
endif()

# Apply flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${BASE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${BASE_CXX_FLAGS_DEBUG}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${BASE_CXX_FLAGS_RELEASE}")

# Enable logging if requested
if(HPC_ENABLE_LOGGING)
    add_definitions(-DHPC_LOGGING_ENABLED)
    
    if(HPC_COLORED_LOGGING)
        add_definitions(-DHPC_COLORED_LOGGING)
    endif()
endif()

# ===================================================================
# CUDA Configuration
# ===================================================================
if(HPC_ENABLE_CUDA)
    # Manually set CUDA architectures instead of auto-detection
    set(CMAKE_CUDA_ARCHITECTURES "70")
    message(STATUS "Using CUDA architecture: ${CMAKE_CUDA_ARCHITECTURES}")
    
    # CUDA compiler flags
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
    set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3 --use_fast_math")
    
    # Separate host and device link flags
    set(CMAKE_CUDA_DEVICE_LINK_FLAGS "${CMAKE_CUDA_DEVICE_LINK_FLAGS} -dlto")
    set(CMAKE_CUDA_HOST_LINK_FLAGS "${CMAKE_CUDA_HOST_LINK_FLAGS} -lpthread -lrt -ldl")
    
    add_definitions(-DCUDA_ENABLED)
endif()

# ===================================================================
# Find Required Packages
# ===================================================================

# Find OpenMP - specify C++ component only to avoid CUDA OpenMP requirement
find_package(OpenMP REQUIRED COMPONENTS CXX)

# Conditionally find GTest
if(HPC_BUILD_TESTS)
    find_package(GTest REQUIRED)
endif()

# ===================================================================
# Define Source Files
# ===================================================================

# CPU sources
set(CPU_SOURCES
    src/compute/cpp/tensor.cpp
    src/compute/cpp/view.cpp
    src/compute/cpp/kernel.cpp
    src/compute/cpp/blas.cpp
    src/compute/cpp/kernels/conv2d.cpp
)

# CUDA sources
if(HPC_ENABLE_CUDA)
    set(CUDA_SOURCES
        src/compute/cuda/tensor.cu
    )
    
    # Ensure CUDA sources are treated as CUDA files
    set_source_files_properties(${CUDA_SOURCES} PROPERTIES LANGUAGE CUDA)
endif()

# ===================================================================
# Library Targets
# ===================================================================

# Common library target properties
function(set_common_target_properties target_name)
    target_include_directories(${target_name}
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
    )
    set_target_properties(${target_name} PROPERTIES
        POSITION_INDEPENDENT_CODE ON
    )
    target_link_libraries(${target_name} PUBLIC OpenMP::OpenMP_CXX)
endfunction()

# Create CPU library
add_library(tensor_cpu STATIC ${CPU_SOURCES})
set_common_target_properties(tensor_cpu)

# Create CUDA library if enabled
if(HPC_ENABLE_CUDA)
    add_library(tensor_cuda STATIC ${CUDA_SOURCES})
    set_common_target_properties(tensor_cuda)
    target_link_libraries(tensor_cuda PUBLIC tensor_cpu)
    
    # Set CUDA specific properties
    set_target_properties(tensor_cuda PROPERTIES 
        CUDA_SEPARABLE_COMPILATION ON
    )
endif()

# ===================================================================
# Tests
# ===================================================================
if(HPC_BUILD_TESTS)
    enable_testing()

    # Main test executable
    add_executable(test_main test/main.cpp)
    target_include_directories(test_main PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    
    if(HPC_ENABLE_CUDA)
        target_link_libraries(test_main 
            PRIVATE 
                tensor_cpu
                tensor_cuda
                GTest::GTest
                GTest::Main
                OpenMP::OpenMP_CXX
        )
    else()
        target_link_libraries(test_main 
            PRIVATE 
                tensor_cpu
                GTest::GTest
                GTest::Main
                OpenMP::OpenMP_CXX
        )
    endif()

    add_test(NAME MainTests COMMAND test_main)
endif()

# ===================================================================
# Installation
# ===================================================================

# Install libraries
install(TARGETS tensor_cpu
    EXPORT hpc_tensor-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(HPC_ENABLE_CUDA)
    install(TARGETS tensor_cuda
        EXPORT hpc_tensor-targets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
endif()

# Install headers
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Export targets
install(EXPORT hpc_tensor-targets
    FILE hpc_tensor-targets.cmake
    NAMESPACE hpc::
    DESTINATION ${INSTALL_CONFIGDIR}
)

# Create config file
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/hpc_tensor-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/hpc_tensor-config.cmake
    INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)

# Install config files
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/hpc_tensor-config.cmake
    DESTINATION ${INSTALL_CONFIGDIR}
)

# ===================================================================
# Summary
# ===================================================================
message(STATUS "")
message(STATUS "HPC Tensor Library Configuration Summary")
message(STATUS "========================================")
message(STATUS "Version: ${PROJECT_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "")
message(STATUS "CPU Support:")
message(STATUS "  AVX512: ${COMPILER_SUPPORTS_AVX512}")
message(STATUS "  AVX2: ${COMPILER_SUPPORTS_AVX2}")
message(STATUS "  FMA: ${COMPILER_SUPPORTS_FMA}")
message(STATUS "  AVX: ${COMPILER_SUPPORTS_AVX}")
message(STATUS "  SSE4.2: ${COMPILER_SUPPORTS_SSE42}")
message(STATUS "")
message(STATUS "Features:")
message(STATUS "  CUDA: ${HPC_ENABLE_CUDA}")
message(STATUS "  Logging: ${HPC_ENABLE_LOGGING}")
message(STATUS "  Colored logging: ${HPC_COLORED_LOGGING}")
message(STATUS "  Tests: ${HPC_BUILD_TESTS}")
message(STATUS "  Benchmarks: ${HPC_BUILD_BENCHMARKS}")
message(STATUS "")