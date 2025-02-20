cmake_minimum_required(VERSION 3.18)
project(hpc_tensor LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build type configuration
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
endif()

# CUDA support
option(CUDA_ENABLED "Enable CUDA support" OFF)

# Check for CUDA
if(CUDA_ENABLED)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        find_package(CUDAToolkit REQUIRED)
        add_definitions(-DCUDA_ENABLED)
        
        # Set CUDA specific flags
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --default-stream per-thread")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --compiler-options -fPIC")
        
        if(CMAKE_BUILD_TYPE STREQUAL "Release")
            set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
            set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")
        endif()
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

# Define source files
set(CPU_SOURCES
    src/compute/cpp/tensor.cpp
    src/compute/cpp/view.cpp
)

set(CUDA_SOURCES
    src/compute/cuda/tensor.cu
)

# Create library
add_library(hpc_tensor ${CPU_SOURCES})

target_include_directories(hpc_tensor
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Configure CUDA sources if enabled
if(CUDA_ENABLED)
    add_library(hpc_tensor_cuda STATIC ${CUDA_SOURCES})
    set_source_files_properties(${CUDA_SOURCES} PROPERTIES LANGUAGE CUDA)
    
    target_include_directories(hpc_tensor_cuda
        PUBLIC
            $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
        PRIVATE
            ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )
    
    set_target_properties(hpc_tensor_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "60;70;75;80"
    )

    target_link_libraries(hpc_tensor PUBLIC hpc_tensor_cuda)
endif()

# Link dependencies
if(OpenMP_CXX_FOUND)
    target_link_libraries(hpc_tensor PUBLIC OpenMP::OpenMP_CXX)
endif()

if(CUDA_ENABLED)
    target_link_libraries(hpc_tensor_cuda PUBLIC CUDA::cudart)
endif()

# Tests
enable_testing()

# Regular CPU tests
add_executable(tensor_test_cpu test/main.cpp)
target_link_libraries(tensor_test_cpu PRIVATE hpc_tensor GTest::GTest GTest::Main)
add_test(NAME tensor_tests_cpu COMMAND tensor_test_cpu)

# CUDA tests if enabled
if(CUDA_ENABLED)
    add_executable(tensor_test_cuda test/main.cpp)
    set_source_files_properties(test/main.cpp PROPERTIES LANGUAGE CUDA)
    target_compile_definitions(tensor_test_cuda PRIVATE CUDA_ENABLED)
    target_link_libraries(tensor_test_cuda PRIVATE hpc_tensor GTest::GTest GTest::Main)
    add_test(NAME tensor_tests_cuda COMMAND tensor_test_cuda)
endif()

# Install rules
include(GNUInstallDirs)
install(TARGETS hpc_tensor
    EXPORT hpc_tensor-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

if(CUDA_ENABLED)
    install(TARGETS hpc_tensor_cuda
        EXPORT hpc_tensor-targets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
endif()

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp" PATTERN "*.cuh"
)

install(EXPORT hpc_tensor-targets
    FILE hpc_tensor-targets.cmake
    NAMESPACE hpc::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hpc_tensor
)