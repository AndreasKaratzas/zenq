# Benchmark CMakeLists.txt
find_package(benchmark REQUIRED)

# Find OpenMP again in this scope
find_package(OpenMP REQUIRED COMPONENTS CXX)

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
        cuda/kernel_bench.cpp
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

# Link with libraries
if(HPC_ENABLE_CUDA)
    target_link_libraries(run_benchmarks
        PRIVATE
            tensor_cpu
            tensor_cuda
            benchmark::benchmark
            benchmark::benchmark_main
            OpenMP::OpenMP_CXX
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