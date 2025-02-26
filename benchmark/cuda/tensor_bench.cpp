#include <benchmark/benchmark.h>

#ifdef CUDA_ENABLED
    #include "compute/cpp/tensor.hpp"
    #include "compute/cuda/tensor.cuh"

    #ifdef HPC_LOGGING_ENABLED
        #include "common/logging.hpp"
    #endif

using namespace hpc::compute;
using namespace hpc::logging;

// Benchmark CUDA tensor creation
static void BM_CUDATensorCreation(benchmark::State& state) {
    const size_t size = state.range(0);

    for (auto _ : state) {
        // This code gets timed
        cuda::Tensor<float> tensor({size, size});
        benchmark::DoNotOptimize(tensor);
    }

    // Set custom counters for items processed and bandwidth
    state.SetItemsProcessed(state.iterations() * size * size);
    state.SetBytesProcessed(state.iterations() * size * size * sizeof(float));
}

// Benchmark host-to-device transfer
static void BM_HostToDeviceTransfer(benchmark::State& state) {
    const size_t size = state.range(0);

    // Create host tensor
    Tensor<float> host_tensor({size, size}, MemoryLayout::RowMajor);

    // Initialize host tensor with random values
    for (size_t i = 0; i < host_tensor.size(); ++i) {
        host_tensor.data()[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    for (auto _ : state) {
        // Benchmark host-to-device transfer
        cuda::Tensor<float> device_tensor = cuda::Tensor<float>::from_host(host_tensor);
        benchmark::DoNotOptimize(device_tensor);
    }

    // Calculate transfer rate
    const size_t bytes_transferred = size * size * sizeof(float);
    state.SetBytesProcessed(state.iterations() * bytes_transferred);

    // Report transfer size in MB
    state.counters["TransferSize_MB"] =
        benchmark::Counter(static_cast<double>(bytes_transferred) / (1024 * 1024));
}

// Benchmark device-to-host transfer
static void BM_DeviceToHostTransfer(benchmark::State& state) {
    const size_t size = state.range(0);

    // Create host and device tensors
    Tensor<float>       host_tensor({size, size}, MemoryLayout::RowMajor);
    cuda::Tensor<float> device_tensor({size, size});

    // Initialize device tensor
    device_tensor.fill(1.0f);

    for (auto _ : state) {
        // Benchmark device-to-host transfer
        host_tensor = device_tensor.to_host();
        benchmark::DoNotOptimize(host_tensor);
        benchmark::ClobberMemory();
    }

    // Calculate transfer rate
    const size_t bytes_transferred = size * size * sizeof(float);
    state.SetBytesProcessed(state.iterations() * bytes_transferred);

    // Report transfer size in MB
    state.counters["TransferSize_MB"] =
        benchmark::Counter(static_cast<double>(bytes_transferred) / (1024 * 1024));
}

// Register benchmarks with different sizes
BENCHMARK(BM_CUDATensorCreation)
    ->RangeMultiplier(2)
    ->Range(64, 4096)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_HostToDeviceTransfer)
    ->RangeMultiplier(2)
    ->Range(64, 4096)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_DeviceToHostTransfer)
    ->RangeMultiplier(2)
    ->Range(64, 4096)
    ->Unit(benchmark::kMillisecond);

#endif // CUDA_ENABLED