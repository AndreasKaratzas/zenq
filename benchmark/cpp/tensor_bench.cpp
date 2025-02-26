#include "compute/cpp/tensor.hpp"
#include <benchmark/benchmark.h>

using namespace hpc::compute;

// Benchmark tensor creation
static void BM_TensorCreation(benchmark::State& state) {
    const size_t size = state.range(0);

    for (auto _ : state) {
        // This code gets timed
        Tensor<float> tensor({size, size}, MemoryLayout::RowMajor);
        benchmark::DoNotOptimize(tensor);
    }

    // Set custom counters for items processed and bandwidth
    state.SetItemsProcessed(state.iterations() * size * size);
    state.SetBytesProcessed(state.iterations() * size * size * sizeof(float));
}

// Benchmark tensor element access
static void BM_TensorElementAccess(benchmark::State& state) {
    const size_t  size = state.range(0);
    Tensor<float> tensor({size, size}, MemoryLayout::RowMajor);

    // Initialize tensor with values
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < size; ++j) {
            tensor(i, j) = static_cast<float>(i * size + j);
        }
    }

    float sum = 0.0f;
    for (auto _ : state) {
        // Reset sum to avoid compiler optimizations
        sum = 0.0f;

        // Benchmark element access
        for (size_t i = 0; i < size; ++i) {
            for (size_t j = 0; j < size; ++j) {
                sum += tensor(i, j);
            }
        }

        benchmark::DoNotOptimize(sum);
    }

    // To prevent compiler optimizations
    benchmark::DoNotOptimize(sum);

    // Set custom counters
    state.SetItemsProcessed(state.iterations() * size * size);
    state.SetBytesProcessed(state.iterations() * size * size * sizeof(float));
}

// Benchmark tensor memory operations
static void BM_TensorZeroOperation(benchmark::State& state) {
    const size_t  size = state.range(0);
    Tensor<float> tensor({size, size}, MemoryLayout::RowMajor);

    for (auto _ : state) {
        // Benchmark zeroing the tensor
        tensor.zero();
        benchmark::ClobberMemory(); // Ensure the operation is not optimized away
    }

    // Set custom counters
    state.SetItemsProcessed(state.iterations() * size * size);
    state.SetBytesProcessed(state.iterations() * size * size * sizeof(float));
}

// Benchmark tensor reshape operations
static void BM_TensorReshape(benchmark::State& state) {
    const size_t  size = state.range(0);
    Tensor<float> tensor({size, size}, MemoryLayout::RowMajor);

    for (auto _ : state) {
        // Benchmark reshape operation
        tensor.reshape({size * size});
        benchmark::DoNotOptimize(tensor);
        tensor.reshape({size, size});
        benchmark::DoNotOptimize(tensor);
    }

    // Set custom counters
    state.SetItemsProcessed(state.iterations() * size * size * 2); // 2 reshape operations
}

// Register benchmarks with different sizes
BENCHMARK(BM_TensorCreation)->RangeMultiplier(2)->Range(64, 4096)->Unit(benchmark::kMillisecond);

BENCHMARK(BM_TensorElementAccess)
    ->RangeMultiplier(2)
    ->Range(64, 2048)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_TensorZeroOperation)
    ->RangeMultiplier(2)
    ->Range(64, 4096)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_TensorReshape)->RangeMultiplier(2)->Range(64, 4096)->Unit(benchmark::kMillisecond);