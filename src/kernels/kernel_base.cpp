#include "backend/compute/engine.hpp"
#include "backend/kernels/cpu_kernel.hpp"
#include "backend/kernels/cuda_kernel.hpp"
#include <stdexcept>

namespace kernel_compute {

Shape KernelBase::compute_output_shape(const Shape& input_shape, const KernelConfig& config) const {
    if (input_shape.size() != 4) {
        throw std::invalid_argument("Input must be 4D (NCHW format)");
    }

    const auto batch_size   = input_shape[0];
    const auto out_channels = config.channels;
    const auto out_height =
        (input_shape[2] - config.kernel_size + 2 * config.padding) / config.stride + 1;
    const auto out_width =
        (input_shape[3] - config.kernel_size + 2 * config.padding) / config.stride + 1;

    return {batch_size, out_channels, out_height, out_width};
}

void KernelBase::validate_inputs(const Tensor&       input,
                                 const Tensor&       weights,
                                 const KernelConfig& config) const {
    if (!config.validate()) {
        throw std::invalid_argument("Invalid kernel configuration");
    }

    if (input.empty() || weights.empty()) {
        throw std::invalid_argument("Input or weights tensor is empty");
    }

    if (input.dtype() != weights.dtype()) {
        throw std::invalid_argument("Input and weights must have the same data type");
    }

    const auto& input_shape  = input.shape();
    const auto& weight_shape = weights.shape();

    if (input_shape.size() != 4 || weight_shape.size() != 4) {
        throw std::invalid_argument("Input and weights must be 4D tensors");
    }

    if (weight_shape[0] != config.channels || weight_shape[2] != config.kernel_size ||
        weight_shape[3] != config.kernel_size) {
        throw std::invalid_argument("Weight dimensions don't match configuration");
    }
}

std::unique_ptr<KernelBase> KernelBase::create(DeviceType device, KernelType type) {
    switch (device) {
    case DeviceType::CPU:
        return type == KernelType::Dense ? std::make_unique<CpuDenseKernel>()
                                         : std::make_unique<CpuSparseKernel>();
    case DeviceType::CUDA:
        return type == KernelType::Dense ? std::make_unique<CudaDenseKernel>()
                                         : std::make_unique<CudaSparseKernel>();
    default:
        throw std::runtime_error("Unsupported device type");
    }
}

} // namespace kernel_compute