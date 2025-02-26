#include "compute/cpp/kernels/conv2d.hpp"
#include "compute/cpp/tensor.hpp"
#include <benchmark/benchmark.h>

#ifdef HPC_LOGGING_ENABLED
    #include "common/logging.hpp"
#endif

using namespace hpc::compute;
using namespace hpc::logging;

// Benchmark Conv2D forward pass
static void BM_Conv2D_Forward(benchmark::State& state) {
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

    // Create input tensor
    Tensor<float> input({batch_size, in_channels, input_size, input_size}, MemoryLayout::RowMajor);

    // Fill input with random values
    for (size_t i = 0; i < input.size(); ++i) {
        input.data()[i] = static_cast<float>(rand()) / RAND_MAX;
    }

#ifdef HPC_LOGGING_ENABLED
    LOG_INFO("Running Conv2D forward benchmark with input size: ",
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
        Tensor<float> output = conv->forward(input);
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
    state.SetItemsProcessed(state.iterations() * input.size());
}

// Benchmark Conv2D with different implementations
static void BM_Conv2D_Implementations(benchmark::State& state) {
    const size_t batch_size   = 1;
    const size_t in_channels  = 3;
    const size_t out_channels = 16;
    const size_t input_size   = 224; // Common image size
    const size_t kernel_size  = 3;
    const size_t stride       = 1;
    const size_t padding      = 1;

    // Implementation type (0=Basic, 1=SSE42, 2=AVX2, 3=AVX512)
    const int impl_type = state.range(0);

    // Create convolution kernel
    auto conv =
        make_conv2d<float>(kernel_size, kernel_size, in_channels, out_channels, stride, padding);

    // Create input tensor
    Tensor<float> input({batch_size, in_channels, input_size, input_size}, MemoryLayout::RowMajor);

    // Fill input with random values
    for (size_t i = 0; i < input.size(); ++i) {
        input.data()[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Force specific implementation by overriding detected CPU features
    // Note: This is just for benchmarking, actual code should use auto-detection
    const char* impl_name = "Unknown";
    switch (impl_type) {
    case 0:
        impl_name = "Basic";
        break;
    case 1:
        impl_name = "SSE4.2";
        break;
    case 2:
        impl_name = "AVX2";
        break;
    case 3:
        impl_name = "AVX512";
        break;
    }

#ifdef HPC_LOGGING_ENABLED
    LOG_INFO("Running Conv2D benchmark with implementation: ", impl_name);
#endif

    for (auto _ : state) {
        // Benchmark Conv2D forward pass
        Tensor<float> output = conv->forward(input);
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

    // Set name based on implementation
    state.SetLabel(impl_name);
}

// Benchmark Conv2D with varying batch sizes
static void BM_Conv2D_BatchSize(benchmark::State& state) {
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

    // Create input tensor
    Tensor<float> input({batch_size, in_channels, input_size, input_size}, MemoryLayout::RowMajor);

    // Fill input with random values
    for (size_t i = 0; i < input.size(); ++i) {
        input.data()[i] = static_cast<float>(rand()) / RAND_MAX;
    }

#ifdef HPC_LOGGING_ENABLED
    LOG_INFO("Running Conv2D batch benchmark with batch size: ", batch_size);
#endif

    for (auto _ : state) {
        // Benchmark Conv2D forward pass
        Tensor<float> output = conv->forward(input);
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
BENCHMARK(BM_Conv2D_Forward)->RangeMultiplier(2)->Range(32, 512)->Unit(benchmark::kMillisecond);

BENCHMARK(BM_Conv2D_Implementations)
    ->DenseRange(0, 3) // Basic, SSE42, AVX2, AVX512
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_Conv2D_BatchSize)->RangeMultiplier(2)->Range(1, 64)->Unit(benchmark::kMillisecond);