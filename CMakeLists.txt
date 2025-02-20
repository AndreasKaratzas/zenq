cmake_minimum_required(VERSION 3.20)
project(hpc_tensor CUDA CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Option to enable/disable CUDA
option(HPC_ENABLE_CUDA "Enable CUDA support" ON)

# Find required packages
find_package(GTest REQUIRED)
find_package(OpenMP REQUIRED)

# Define source files
set(CPU_SOURCES
    src/compute/cpp/tensor.cpp
    src/compute/cpp/view.cpp
)

if(HPC_ENABLE_CUDA)
    set(CUDA_SOURCES
        src/compute/cuda/tensor.cu
    )
    set(CMAKE_CUDA_ARCHITECTURES 75)  # Adjust based on your GPU
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
    
    # Separate host and device link flags
    set(CMAKE_CUDA_DEVICE_LINK_FLAGS "${CMAKE_CUDA_DEVICE_LINK_FLAGS} -dlto")
    set(CMAKE_CUDA_HOST_LINK_FLAGS "${CMAKE_CUDA_HOST_LINK_FLAGS} -lpthread -lrt -ldl")
    
    add_definitions(-DCUDA_ENABLED)
    
    # Ensure CUDA sources are treated as CUDA files
    set_source_files_properties(${CUDA_SOURCES} PROPERTIES LANGUAGE CUDA)
endif()

# Create CPU library
add_library(tensor_cpu STATIC ${CPU_SOURCES})
target_include_directories(tensor_cpu
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Create CUDA library if enabled
if(HPC_ENABLE_CUDA)
    add_library(tensor_cuda STATIC ${CUDA_SOURCES})
    target_include_directories(tensor_cuda
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
    )
    target_link_libraries(tensor_cuda PUBLIC tensor_cpu)
    
    # Set CUDA specific properties
    set_target_properties(tensor_cuda PROPERTIES 
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
    )
endif()

# Tests
enable_testing()

# Main test executable
add_executable(test_main test/main.cpp)
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

# Installation rules
include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/hpc_tensor)

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

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Export targets
install(EXPORT hpc_tensor-targets
    FILE hpc_tensor-targets.cmake
    NAMESPACE hpc::
    DESTINATION ${INSTALL_CONFIGDIR}
)

# Create config file
include(CMakePackageConfigHelpers)
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/hpc_tensor-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/hpc_tensor-config.cmake
    INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/hpc_tensor-config.cmake
    DESTINATION ${INSTALL_CONFIGDIR}
)