#pragma once

#include "backend/compute/kernel_base.hpp"
#include "tensor.hpp"
#include <memory>

namespace hpc {
namespace compute {

class Engine {
public:
    explicit Engine(hpc::core::DeviceType device_type);

    // High-level operations
    Tensor conv2d(const Tensor&                  input,
                  const Tensor&                  weights,
                  const hpc::config::ConvConfig& config);

private:
    std::unique_ptr<KernelBase> kernel_;
    hpc::core::DeviceType       device_type_;
};

} // namespace compute
} // namespace hpc