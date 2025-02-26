#include <benchmark/benchmark.h>

#ifdef CUDA_ENABLED
    #include "compute/cpp/tensor.hpp"
    #include "compute/cuda/tensor.cuh"

    #ifdef HPC_LOGGING_ENABLED
        #include "common/logging.hpp"
    #endif

using namespace hpc::compute;
using namespace hpc::logging;

// Utility function to convert cuda error enum to string
const char* cudaGetErrorString(int error);

// Benchmark CUDA matrix multiplication
static void BM_CUDAMatrixMultiplication(benchmark::State& state) {
    const size_t size = state.range(0);

    // Create CUDA tensors
    cuda::Tensor<float> A({size, size});
    cuda::Tensor<float> B({size, size});
    cuda::Tensor<float> C({size, size});

    // Initialize with random values
    A.fill(1.0f);
    B.fill(1.0f);

    #ifdef HPC_LOGGING_ENABLED
    LOG_INFO("Running CUDA GEMM benchmark with size: ", size, "x", size);
    #endif

    for (auto _ : state) {
        // Normally, you would call the CUDA implementation of GEMM here
        // For demonstration purposes, this is a placeholder
        // In real implementation, replace this with your actual CUDA kernel call

        // Get the start time
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        // Call your CUDA kernel or operation here
        // cuda::gemm(false, false, 1.0f, A, B, 0.0f, C);

        // Get the end time
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        // Cleanup events
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        // Manually set the iteration time
        state.SetIterationTime(milliseconds / 1000.0);

        // Ensure synchronization
        cudaDeviceSynchronize();
        benchmark::ClobberMemory();
    }

    // Calculate theoretical FLOPs (2*n*n*n for gemm)
    double flops            = 2.0 * size * size * size;
    state.counters["FLOPS"] = benchmark::Counter(
        flops, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1000);

    // Report matrix size
    state.counters["MatrixSize"] = static_cast<double>(size);
}

// Benchmark CUDA element-wise operations
static void BM_CUDAElementWiseOps(benchmark::State& state) {
    const size_t size = state.range(0);

    // Create CUDA tensors
    cuda::Tensor<float> A({size, size});
    cuda::Tensor<float> B({size, size});
    cuda::Tensor<float> C({size, size});

    // Initialize with values
    A.fill(1.5f);
    B.fill(2.5f);

    // The operation type (0=add, 1=multiply, 2=divide)
    const int   op_type = state.range(1);
    const char* op_name = "Unknown";

    switch (op_type) {
    case 0:
        op_name = "Add";
        break;
    case 1:
        op_name = "Multiply";
        break;
    case 2:
        op_name = "Divide";
        break;
    }

    #ifdef HPC_LOGGING_ENABLED
    LOG_INFO("Running CUDA element-wise ", op_name, " benchmark with size: ", size, "x", size);
    #endif

    for (auto _ : state) {
        // Get the start time
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        // Call the appropriate CUDA operation
        switch (op_type) {
        case 0:
            // C = A + B
            // cuda::add(A, B, C);
            break;
        case 1:
            // C = A * B
            // cuda::multiply(A, B, C);
            break;
        case 2:
            // C = A / B
            // cuda::divide(A, B, C);
            break;
        }

        // Get the end time
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        // Cleanup events
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        // Manually set the iteration time
        state.SetIterationTime(milliseconds / 1000.0);

        // Ensure synchronization
        cudaDeviceSynchronize();
        benchmark::ClobberMemory();
    }

    // Calculate throughput in elements per second
    const size_t num_elements = size * size;
    state.SetItemsProcessed(state.iterations() * num_elements);
    state.SetBytesProcessed(state.iterations() * num_elements * sizeof(float) *
                            3); // 2 inputs, 1 output

    // Set label based on operation
    state.SetLabel(op_name);
}

// Register benchmarks with different sizes
BENCHMARK(BM_CUDAMatrixMultiplication)
    ->RangeMultiplier(2)
    ->Range(512, 4096)
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_CUDAElementWiseOps)
    ->Args({1024, 0}) // Size 1024, operation Add
    ->Args({1024, 1}) // Size 1024, operation Multiply
    ->Args({1024, 2}) // Size 1024, operation Divide
    ->Args({2048, 0})
    ->Args({2048, 1})
    ->Args({2048, 2})
    ->Args({4096, 0})
    ->Args({4096, 1})
    ->Args({4096, 2})
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond);

#endif // CUDA_ENABLED