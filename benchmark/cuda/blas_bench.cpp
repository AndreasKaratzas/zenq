#include <benchmark/benchmark.h>

#ifdef CUDA_ENABLED
    #include "compute/cpp/tensor.hpp"
    #include "compute/cuda/blas.cuh"
    #include "compute/cuda/tensor.cuh"
    #include "compute/cuda/wrapper.hpp"

    #ifdef HPC_LOGGING_ENABLED
        #include "common/logging.hpp"
    #endif

using namespace hpc::compute;
using namespace hpc::compute::cuda;
using namespace hpc::logging;

// Benchmark CUDA GEMM operation
static void BM_CUDA_GEMM(benchmark::State& state) {
    const size_t m = state.range(0);
    const size_t n = state.range(0);
    const size_t k = state.range(0);

    // Initialize matrices on CPU
    Tensor<float> host_A({m, k}, hpc::compute::MemoryLayout::RowMajor);
    Tensor<float> host_B({k, n}, hpc::compute::MemoryLayout::RowMajor);
    Tensor<float> host_C({m, n}, hpc::compute::MemoryLayout::RowMajor);

    // Fill matrices with random values
    for (size_t i = 0; i < m * k; ++i) {
        host_A.data()[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (size_t i = 0; i < k * n; ++i) {
        host_B.data()[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Transfer to GPU
    TensorWrapper<float> A(host_A);
    TensorWrapper<float> B(host_B);
    TensorWrapper<float> C({m, n});

    // Initialize BLAS
    BLAS<float>::initialize();

    #ifdef HPC_LOGGING_ENABLED
    LOG_INFO("Running CUDA GEMM benchmark with matrix size: ", m, "x", k, " * ", k, "x", n);
    #endif

    for (auto _ : state) {
        // Benchmark GEMM operation
        BLAS<float>::gemm(false, false, 1.0f, A, B, 0.0f, C);
        cudaDeviceSynchronize(); // Ensure operation completes before timing
        benchmark::ClobberMemory();
    }

    // Calculate FLOPs (2*m*n*k for gemm)
    double flops            = 2.0 * m * n * k;
    state.counters["FLOPS"] = benchmark::Counter(
        flops, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1000);

    // Set bytes processed (input matrices + output matrix)
    state.SetBytesProcessed(state.iterations() * (m * k + k * n + m * n) * sizeof(float));

    // Clean up
    BLAS<float>::finalize();
}

// Benchmark im2col operation
static void BM_CUDA_Im2Col(benchmark::State& state) {
    const size_t batch_size  = 1;
    const size_t channels    = 3;
    const size_t height      = state.range(0);
    const size_t width       = state.range(0);
    const size_t kernel_size = 3;
    const size_t padding     = 1;
    const size_t stride      = 1;

    // Calculate output dimensions
    const size_t output_height = (height + 2 * padding - kernel_size) / stride + 1;
    const size_t output_width  = (width + 2 * padding - kernel_size) / stride + 1;

    // Initialize tensors on CPU
    Tensor<float> host_input({batch_size, channels, height, width},
                             hpc::compute::MemoryLayout::RowMajor);

    // Fill input with random values
    for (size_t i = 0; i < host_input.size(); ++i) {
        host_input.data()[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Transfer to GPU
    TensorWrapper<float> input(host_input);
    TensorWrapper<float> col_buffer(
        {batch_size, channels * kernel_size * kernel_size, output_height * output_width});

    #ifdef HPC_LOGGING_ENABLED
    LOG_INFO("Running CUDA im2col benchmark with input size: ",
             batch_size,
             "x",
             channels,
             "x",
             height,
             "x",
             width,
             " kernel: ",
             kernel_size,
             "x",
             kernel_size);
    #endif

    for (auto _ : state) {
        // Benchmark im2col operation
        BLAS<float>::im2col(
            input, kernel_size, kernel_size, padding, padding, stride, stride, col_buffer);
        cudaDeviceSynchronize();
        benchmark::ClobberMemory();
    }

    // Set items processed
    state.SetItemsProcessed(state.iterations() * batch_size * channels * height * width);

    // Set bytes processed (input tensor + col_buffer)
    state.SetBytesProcessed(state.iterations() * (host_input.size() + col_buffer.size()) *
                            sizeof(float));
}

// Benchmark col2im operation
static void BM_CUDA_Col2Im(benchmark::State& state) {
    const size_t batch_size  = 1;
    const size_t channels    = 3;
    const size_t height      = state.range(0);
    const size_t width       = state.range(0);
    const size_t kernel_size = 3;
    const size_t padding     = 1;
    const size_t stride      = 1;

    // Calculate output dimensions
    const size_t output_height = (height + 2 * padding - kernel_size) / stride + 1;
    const size_t output_width  = (width + 2 * padding - kernel_size) / stride + 1;

    // Initialize tensors on CPU
    Tensor<float> host_col_buffer(
        {batch_size, channels * kernel_size * kernel_size, output_height * output_width},
        hpc::compute::MemoryLayout::RowMajor);

    // Fill col_buffer with random values
    for (size_t i = 0; i < host_col_buffer.size(); ++i) {
        host_col_buffer.data()[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Transfer to GPU
    TensorWrapper<float> col_buffer(host_col_buffer);
    TensorWrapper<float> output({batch_size, channels, height, width});

    #ifdef HPC_LOGGING_ENABLED
    LOG_INFO("Running CUDA col2im benchmark with col_buffer size: ",
             batch_size,
             "x",
             channels * kernel_size * kernel_size,
             "x",
             output_height * output_width);
    #endif

    for (auto _ : state) {
        // Benchmark col2im operation
        BLAS<float>::col2im(col_buffer,
                            channels,
                            height,
                            width,
                            kernel_size,
                            kernel_size,
                            padding,
                            padding,
                            stride,
                            stride,
                            output);
        cudaDeviceSynchronize();
        benchmark::ClobberMemory();
    }

    // Set items processed
    state.SetItemsProcessed(state.iterations() * host_col_buffer.size());

    // Set bytes processed (col_buffer + output tensor)
    state.SetBytesProcessed(state.iterations() * (output.size() + col_buffer.size()) *
                            sizeof(float));
}

// Register benchmarks with different sizes
BENCHMARK(BM_CUDA_GEMM)->RangeMultiplier(2)->Range(128, 2048)->Unit(benchmark::kMillisecond);

BENCHMARK(BM_CUDA_Im2Col)->RangeMultiplier(2)->Range(32, 512)->Unit(benchmark::kMillisecond);

BENCHMARK(BM_CUDA_Col2Im)->RangeMultiplier(2)->Range(32, 512)->Unit(benchmark::kMillisecond);

#endif // CUDA_ENABLED