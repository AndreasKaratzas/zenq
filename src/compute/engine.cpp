#include "backend/compute/engine.hpp"
#include "backend/kernels/cpu/conv_kernel.hpp"
#include "backend/kernels/cuda/conv_kernel.cuh"

namespace kernel_compute {

std::unique_ptr<Engine> Engine::create(DeviceType device) {
    switch (device) {
    case DeviceType::CPU:
        return std::make_unique<CpuEngine>();
    case DeviceType::CUDA:
        return std::make_unique<CudaEngine>();
    default:
        throw std::runtime_error("Unsupported device type");
    }
}

} // namespace kernel_compute