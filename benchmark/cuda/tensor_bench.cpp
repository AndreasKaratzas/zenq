#include <benchmark/benchmark.h>

#ifdef CUDA_ENABLED
    #include "compute/cpp/tensor.hpp"
    #include "compute/cuda/tensor.cuh"
    #include "compute/cuda/wrapper.hpp"

using namespace hpc::compute;

// Benchmark CUDA tensor creation
static void BM_CUDATensorCreation(benchmark::State& state) {
    const size_t size = state.range(0);

    for (auto _ : state) {
        // This code gets timed
        cuda::TensorWrapper<float> tensor({size, size});
        benchmark::DoNotOptimize(tensor);
    }

    // Set custom counters for items processed and bandwidth
    state.SetItemsProcessed(state.iterations() * size * size);
    state.SetBytesProcessed(state.iterations() * size * size * sizeof(float));
}

// Benchmark host-to-device transfer
static void BM_HostToDeviceTransfer(benchmark::State& state) {
    const size_t size = state.range(0);

    // Create host tensor with initialized data
    Tensor<float> host_tensor({size, size}, MemoryLayout::RowMajor);
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < size; ++j) {
            host_tensor(i, j) = static_cast<float>(i * size + j);
        }
    }

    // Create device tensor
    cuda::TensorWrapper<float> device_tensor({size, size});

    for (auto _ : state) {
        // Benchmark host-to-device transfer
        device_tensor.copy_from_host(host_tensor);
        benchmark::ClobberMemory();
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
    Tensor<float>              host_tensor({size, size}, MemoryLayout::RowMajor);
    cuda::TensorWrapper<float> device_tensor({size, size});

    // Initialize device tensor with values from host
    Tensor<float> init_tensor({size, size});
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < size; ++j) {
            init_tensor(i, j) = static_cast<float>(i * size + j);
        }
    }
    device_tensor.copy_from_host(init_tensor);

    for (auto _ : state) {
        // Benchmark device-to-host transfer
        device_tensor.copy_to_host(host_tensor);
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

// Benchmark zero operation
static void BM_CUDATensorZeroOperation(benchmark::State& state) {
    const size_t size = state.range(0);

    // Create device tensor
    cuda::TensorWrapper<float> tensor({size, size});

    for (auto _ : state) {
        // Benchmark zeroing the tensor
        tensor.zero();
        benchmark::ClobberMemory();
    }

    // Set custom counters
    state.SetItemsProcessed(state.iterations() * size * size);
    state.SetBytesProcessed(state.iterations() * size * size * sizeof(float));
}

// Benchmark layout conversion
static void BM_CUDALayoutConversion(benchmark::State& state) {
    const size_t size = state.range(0);

    // Create row-major host tensor with initialized data
    Tensor<float> host_tensor({size, size}, MemoryLayout::RowMajor);
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < size; ++j) {
            host_tensor(i, j) = static_cast<float>(i * size + j);
        }
    }

    // Create column-major device tensor
    cuda::TensorWrapper<float> device_tensor({size, size}, cuda::MemoryLayout::ColumnMajor);

    // Result tensor for copy back
    Tensor<float> result_tensor({size, size}, MemoryLayout::ColumnMajor);

    for (auto _ : state) {
        // Benchmark layout conversion (row-major to column-major and back)
        device_tensor.copy_from_host(host_tensor);
        device_tensor.copy_to_host(result_tensor);
        benchmark::ClobberMemory();
    }

    // Calculate bytes processed
    const size_t bytes_processed = 2 * size * size * sizeof(float); // Two transfers
    state.SetBytesProcessed(state.iterations() * bytes_processed);
}

// Benchmark move operations
static void BM_CUDATensorMove(benchmark::State& state) {
    const size_t size = state.range(0);

    for (auto _ : state) {
        // Create a tensor and move it
        cuda::TensorWrapper<float> tensor1({size, size});

        // Initialize with some data
        Tensor<float> host_data({size, size});
        for (size_t i = 0; i < host_data.size(); ++i) {
            host_data.data()[i] = static_cast<float>(i);
        }
        tensor1.copy_from_host(host_data);

        // Move construction
        cuda::TensorWrapper<float> tensor2(std::move(tensor1));
        benchmark::DoNotOptimize(tensor2);

        // Verify tensor1 is empty
        benchmark::DoNotOptimize(tensor1.size() == 0);
    }

    // Set custom counters
    state.SetItemsProcessed(state.iterations());
    state.SetBytesProcessed(state.iterations() * size * size * sizeof(float));
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

BENCHMARK(BM_CUDATensorZeroOperation)
    ->RangeMultiplier(2)
    ->Range(64, 4096)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_CUDALayoutConversion)
    ->RangeMultiplier(2)
    ->Range(64, 2048)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_CUDATensorMove)->RangeMultiplier(2)->Range(64, 2048)->Unit(benchmark::kMillisecond);

#endif // CUDA_ENABLED