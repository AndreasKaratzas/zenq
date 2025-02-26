#include <benchmark/benchmark.h>

#ifdef CUDA_ENABLED
    #include "compute/cpp/tensor.hpp"
    #include "compute/cuda/kernels/conv2d.cuh"
    #include "compute/cuda/tensor.cuh"
    #include "compute/cuda/wrapper.hpp"

    #ifdef HPC_LOGGING_ENABLED
        #include "common/logging.hpp"
    #endif

using namespace hpc::compute;
using namespace hpc::compute::cuda;
using namespace hpc::logging;

// Benchmark Conv2D forward pass
static void BM_CUDA_Conv2D_Forward(benchmark::State& state) {
    const size_t batch_size   = 1;
    const size_t in_channels  = 3;
    const size_t out_channels = 16;
    const size_t input_size   = state.range(0);
    const size_t kernel_size  = 3;
    const size_t stride       = 1;
    const size_t padding      = 1;

    // Create convolution kernel
    auto conv =
        make_conv2d<float>(kernel_size, kernel_size, in_channels, out_channels, stride, padding);

    // Create input tensor on CPU
    Tensor<float> host_input({batch_size, in_channels, input_size, input_size},
                             hpc::compute::MemoryLayout::RowMajor);

    // Fill input with random values
    for (size_t i = 0; i < host_input.size(); ++i) {
        host_input.data()[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    #ifdef HPC_LOGGING_ENABLED
    LOG_INFO("Running CUDA Conv2D forward benchmark with input size: ",
             batch_size,
             "x",
             in_channels,
             "x",
             input_size,
             "x",
             input_size,
             " kernel: ",
             kernel_size,
             "x",
             kernel_size,
             " out_channels: ",
             out_channels);
    #endif

    for (auto _ : state) {
        // Benchmark Conv2D forward pass
        Tensor<float> output = conv->forward(host_input);
        cudaDeviceSynchronize();
        benchmark::DoNotOptimize(output);
        benchmark::ClobberMemory();
    }

    // Calculate theoretical FLOPs
    // For each output element: in_channels * kernel_size * kernel_size multiplications and
    // additions
    const size_t output_size      = input_size; // Same with padding=1, stride=1, kernel=3
    const size_t flops_per_output = 2 * in_channels * kernel_size * kernel_size; // mul + add
    const size_t total_flops =
        batch_size * out_channels * output_size * output_size * flops_per_output;

    state.counters["FLOPS"] = benchmark::Counter(static_cast<double>(total_flops),
                                                 benchmark::Counter::kIsIterationInvariantRate,
                                                 benchmark::Counter::OneK::kIs1000);

    // Set items processed (number of input elements)
    state.SetItemsProcessed(state.iterations() * host_input.size());
}

// Benchmark Conv2D with varying batch sizes
static void BM_CUDA_Conv2D_BatchSize(benchmark::State& state) {
    const size_t batch_size   = state.range(0);
    const size_t in_channels  = 3;
    const size_t out_channels = 16;
    const size_t input_size   = 112; // Smaller image for larger batches
    const size_t kernel_size  = 3;
    const size_t stride       = 1;
    const size_t padding      = 1;

    // Create convolution kernel
    auto conv =
        make_conv2d<float>(kernel_size, kernel_size, in_channels, out_channels, stride, padding);

    // Create input tensor on CPU
    Tensor<float> host_input({batch_size, in_channels, input_size, input_size},
                             hpc::compute::MemoryLayout::RowMajor);

    // Fill input with random values
    for (size_t i = 0; i < host_input.size(); ++i) {
        host_input.data()[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    #ifdef HPC_LOGGING_ENABLED
    LOG_INFO("Running CUDA Conv2D batch benchmark with batch size: ", batch_size);
    #endif

    for (auto _ : state) {
        // Benchmark Conv2D forward pass
        Tensor<float> output = conv->forward(host_input);
        cudaDeviceSynchronize();
        benchmark::DoNotOptimize(output);
        benchmark::ClobberMemory();
    }

    // Calculate theoretical FLOPs
    const size_t output_size      = input_size;
    const size_t flops_per_output = 2 * in_channels * kernel_size * kernel_size;
    const size_t total_flops =
        batch_size * out_channels * output_size * output_size * flops_per_output;

    state.counters["FLOPS"] = benchmark::Counter(static_cast<double>(total_flops),
                                                 benchmark::Counter::kIsIterationInvariantRate,
                                                 benchmark::Counter::OneK::kIs1000);

    // Calculate efficiency (normalized to batch size)
    state.counters["Efficiency"] = benchmark::Counter(static_cast<double>(total_flops) / batch_size,
                                                      benchmark::Counter::kIsIterationInvariantRate,
                                                      benchmark::Counter::OneK::kIs1000);
}

// Register benchmarks
BENCHMARK(BM_CUDA_Conv2D_Forward)
    ->RangeMultiplier(2)
    ->Range(32, 512)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_CUDA_Conv2D_BatchSize)
    ->RangeMultiplier(2)
    ->Range(1, 64)
    ->Unit(benchmark::kMillisecond);

#endif // CUDA_ENABLED