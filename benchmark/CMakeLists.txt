# Benchmark CMakeLists.txt
find_package(benchmark REQUIRED)
# Find OpenMP again in this scope
find_package(OpenMP REQUIRED COMPONENTS CXX)

# Find CUDA
if(HPC_ENABLE_CUDA)
    # Use the modern CUDAToolkit module if using CMake 3.17+
    if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.17.0")
        find_package(CUDAToolkit REQUIRED)
    else()
        find_package(CUDA REQUIRED)
        # If using older CMake, we need to set up CUDA::cudart manually
        if(NOT TARGET CUDA::cudart)
            add_library(CUDA::cudart IMPORTED INTERFACE)
            set_target_properties(CUDA::cudart PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${CUDA_INCLUDE_DIRS}"
                INTERFACE_LINK_LIBRARIES "${CUDA_LIBRARIES}"
            )
        endif()
    endif()
endif()

# Set C++17 as the minimum standard (change to C++20 if concepts are needed)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(HPC_ENABLE_CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# Benchmark source files
set(BENCHMARK_CPU_SOURCES
    cpp/tensor_bench.cpp
    cpp/blas_bench.cpp
    cpp/conv2d_bench.cpp
)

# CUDA benchmark sources (only if CUDA is enabled)
if(HPC_ENABLE_CUDA)
    set(BENCHMARK_CUDA_SOURCES
        cuda/tensor_bench.cpp
        cuda/blas_bench.cpp
        cuda/conv2d_bench.cpp
    )
endif()

# Create main benchmark executable
add_executable(run_benchmarks 
    main.cpp
    ${BENCHMARK_CPU_SOURCES}
    ${BENCHMARK_CUDA_SOURCES}
)

# Set include directories
target_include_directories(run_benchmarks 
    PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

if(HPC_ENABLE_CUDA)
    target_compile_definitions(run_benchmarks PRIVATE CUDA_ENABLED)
endif()

# Link with libraries
if(HPC_ENABLE_CUDA)
    target_link_libraries(run_benchmarks
        PRIVATE
            tensor_cpu
            tensor_cuda
            benchmark::benchmark
            benchmark::benchmark_main
            OpenMP::OpenMP_CXX
            CUDA::cudart
    )
else()
    target_link_libraries(run_benchmarks
        PRIVATE
            tensor_cpu
            benchmark::benchmark
            benchmark::benchmark_main
            OpenMP::OpenMP_CXX
    )
endif()

# Set compile definitions based on parent settings
if(HPC_ENABLE_LOGGING)
    target_compile_definitions(run_benchmarks PRIVATE HPC_LOGGING_ENABLED)
    if(HPC_COLORED_LOGGING)
        target_compile_definitions(run_benchmarks PRIVATE HPC_COLORED_LOGGING)
    endif()
endif()

# Set output directory
set_target_properties(run_benchmarks PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)