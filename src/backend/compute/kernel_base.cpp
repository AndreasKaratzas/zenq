#include "backend/compute/kernel_base.hpp"
#include "backend/compute/cpu/conv_kernel.hpp"
#ifdef CUDA_ENABLED
    #include "backend/compute/cuda/conv_kernel.cuh"
#endif
#include <stdexcept>

namespace hpc {
namespace compute {

std::unique_ptr<KernelBase> KernelBase::create(hpc::core::DeviceType device) {
    switch (device) {
    case hpc::core::DeviceType::CPU:
        return std::make_unique<compute::cpu::ConvKernel>();
    case hpc::core::DeviceType::CUDA:
#ifdef CUDA_ENABLED
        return std::make_unique<compute::cuda::ConvKernel>();
#else
        throw std::runtime_error("CUDA support not enabled");
#endif
    default:
        throw std::runtime_error("Unsupported device type");
    }
}

hpc::core::Shape KernelBase::compute_output_shape(const hpc::core::Shape&        input_shape,
                                                  const hpc::config::ConvConfig& config) const {
    if (input_shape.size() != 4) { // N, C, H, W
        throw std::runtime_error("Input must be 4D (N, C, H, W)");
    }

    size_t batch_size      = input_shape[0];
    size_t output_channels = config.out_channels;
    size_t input_height    = input_shape[2];
    size_t input_width     = input_shape[3];

    // Calculate output dimensions considering padding, stride, and dilation
    size_t output_height =
        (input_height + 2 * config.padding - config.dilation * (config.kernel_size - 1) - 1) /
            config.stride +
        1;
    size_t output_width =
        (input_width + 2 * config.padding - config.dilation * (config.kernel_size - 1) - 1) /
            config.stride +
        1;

    return {batch_size, output_channels, output_height, output_width};
}

void KernelBase::validate_inputs(const Tensor&                  input,
                                 const Tensor&                  weights,
                                 const hpc::config::ConvConfig& config) const {
    if (input.dtype() != weights.dtype()) {
        throw std::runtime_error("Input and weights must have the same data type");
    }

    const auto& input_shape  = input.shape();
    const auto& weight_shape = weights.shape();

    if (input_shape.size() != 4 || weight_shape.size() != 4) {
        throw std::runtime_error("Input and weights must be 4D tensors");
    }

    if (input_shape[1] != config.in_channels) {
        throw std::runtime_error("Input channels don't match configuration");
    }

    if (weight_shape !=
        hpc::core::Shape{
            config.out_channels, config.in_channels, config.kernel_size, config.kernel_size}) {
        throw std::runtime_error("Weight shape doesn't match configuration");
    }

    // Validate stride, padding, and dilation
    if (config.stride == 0) {
        throw std::runtime_error("Stride cannot be zero");
    }

    if (config.dilation == 0) {
        throw std::runtime_error("Dilation cannot be zero");
    }

    // Validate that output dimensions will be positive
    auto output_shape = compute_output_shape(input_shape, config);
    if (output_shape[2] <= 0 || output_shape[3] <= 0) {
        throw std::runtime_error(
            "Invalid configuration: output dimensions would be zero or negative");
    }

    // Validate groups if using grouped convolution
    if (config.groups > 1) {
        if (config.in_channels % config.groups != 0 || config.out_channels % config.groups != 0) {
            throw std::runtime_error("Input and output channels must be divisible by groups");
        }
    }
}

} // namespace compute
} // namespace hpc