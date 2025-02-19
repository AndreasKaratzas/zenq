cmake_minimum_required(VERSION 3.18)
project(hpc_tensor LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# CUDA support
option(CUDA_ENABLED "Enable CUDA support" OFF)

# Check for CUDA
if(CUDA_ENABLED)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 14)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        find_package(CUDAToolkit REQUIRED)
        add_definitions(-DCUDA_ENABLED)
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
    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

# Check for AVX support
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
check_cxx_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512F)

if(COMPILER_SUPPORTS_AVX512F)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512f")
elseif(COMPILER_SUPPORTS_AVX2)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
endif()

# Create library
add_library(tensor
    src/backend/common/tensor.cpp
)

target_include_directories(tensor
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

if(CMAKE_CUDA_COMPILER)
    target_link_libraries(tensor
        PUBLIC
            CUDA::cudart
    )
endif()

# Tests
enable_testing()
add_executable(tensor_test 
    test/backend.cpp
)

target_link_libraries(tensor_test
    PRIVATE
        tensor
        GTest::GTest
        GTest::Main
)

# Install rules
include(GNUInstallDirs)
install(TARGETS tensor
    EXPORT tensor-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(FILES
    include/tensor.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/hpc
)

install(EXPORT tensor-targets
    FILE tensor-targets.cmake
    NAMESPACE hpc::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tensor
)