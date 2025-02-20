#include "backend/compute/engine.hpp"

namespace hpc {
namespace compute {

Engine::Engine(hpc::core::DeviceType device_type) : device_type_(device_type) {
    kernel_ = KernelBase::create(device_type);
}

Tensor Engine::conv2d(const Tensor&                  input,
                      const Tensor&                  weights,
                      const hpc::config::ConvConfig& config) {
    return kernel_->forward(input, weights, config);
}

} // namespace compute
} // namespace hpc