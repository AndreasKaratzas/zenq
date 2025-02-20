cmake_minimum_required(VERSION 3.18)
project(hpc_tensor LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# CUDA support
option(CUDA_ENABLED "Enable CUDA support" OFF)

# Add these lines in your CMakeLists.txt where CUDA is configured
if(CUDA_ENABLED)
    # Add CUDA compilation flags
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
    
    # Enable extended lambda features
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")
    
    # Set default stream to per-thread
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --default-stream per-thread")
    
    # Add debug info in release
    set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -g")
    
    # Set maximum register count for better occupancy
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --maxrregcount=64")
endif()

# Check for CUDA
if(CUDA_ENABLED)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)  # CUDA 17 for better C++20 compatibility
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        find_package(CUDAToolkit REQUIRED)
        add_definitions(-DCUDA_ENABLED)
        
        # Set CUDA specific flags
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++17")
    else()
        message(FATAL_ERROR "CUDA compiler not found. Please install CUDA or disable CUDA support.")
    endif()
endif()

# Handle GTest dependency
find_package(GTest QUIET)
if(NOT GTest_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG release-1.12.1
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

# Check for OpenMP
find_package(OpenMP)

# Check for AVX support
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
check_cxx_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512F)

if(COMPILER_SUPPORTS_AVX512F)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512f")
    add_definitions(-D__AVX512F__)
elseif(COMPILER_SUPPORTS_AVX2)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
    add_definitions(-D__AVX2__)
endif()

# Define source files
set(CPU_SOURCES
    src/backend/compute/tensor.cpp
    src/backend/compute/engine.cpp
    src/backend/compute/kernel_base.cpp  # Added this
    src/backend/compute/cpu/conv_kernel.cpp
)

set(CUDA_SOURCES
    src/backend/compute/cuda/conv_kernel.cu
)

# Create library
add_library(hpc_tensor)

# Add CPU sources
target_sources(hpc_tensor
    PRIVATE
        ${CPU_SOURCES}
)

# Configure CUDA sources if enabled
if(CUDA_ENABLED)
    # Set properties for CUDA files
    set_source_files_properties(${CUDA_SOURCES}
        PROPERTIES
        LANGUAGE CUDA
        CUDA_SOURCE_PROPERTY_FORMAT OBJ
        CUDA_SEPARABLE_COMPILATION ON
    )
    
    # Add CUDA sources to the target
    target_sources(hpc_tensor
        PRIVATE
            ${CUDA_SOURCES}
    )
    
    # Configure CUDA target properties
    set_target_properties(hpc_tensor 
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "60;70;75;80"
    )
    
    # Add CUDA include directories
    target_include_directories(hpc_tensor
        PRIVATE
            ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )
endif()

# Add main include directory
target_include_directories(hpc_tensor
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Link dependencies
if(OpenMP_CXX_FOUND)
    target_link_libraries(hpc_tensor PUBLIC OpenMP::OpenMP_CXX)
endif()

if(CUDA_ENABLED)
    target_link_libraries(hpc_tensor
        PUBLIC
            CUDA::cudart
    )
endif()

# Tests
enable_testing()
add_executable(backend_test 
    test/backend.cpp
)

target_link_libraries(backend_test
    PRIVATE
        hpc_tensor
        GTest::GTest
        GTest::Main
)

# Install rules
include(GNUInstallDirs)
install(TARGETS hpc_tensor
    EXPORT hpc_tensor-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install header files maintaining directory structure
install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp" PATTERN "*.cuh"
)

install(EXPORT hpc_tensor-targets
    FILE hpc_tensor-targets.cmake
    NAMESPACE hpc::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hpc_tensor
)

# Generate and install package configuration
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/hpc_tensor-config-version.cmake"
    VERSION 0.1.0
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/hpc_tensor-config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/hpc_tensor-config.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hpc_tensor
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/hpc_tensor-config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/hpc_tensor-config-version.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hpc_tensor
)