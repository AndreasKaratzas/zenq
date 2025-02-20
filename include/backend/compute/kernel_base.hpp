#pragma once

#include "backend/common/config.hpp"
#include "backend/compute/tensor.hpp"
#include <memory>

namespace hpc {
namespace compute {

class KernelBase {
public:
    virtual ~KernelBase() = default;

    // Main computation interface
    virtual Tensor forward(const Tensor&                  input,
                           const Tensor&                  weights,
                           const hpc::config::ConvConfig& config) = 0;

    // Factory method
    static std::unique_ptr<KernelBase> create(hpc::core::DeviceType device);

protected:
    // Utility methods for derived classes
    hpc::core::Shape compute_output_shape(const hpc::core::Shape&        input_shape,
                                          const hpc::config::ConvConfig& config) const;

    virtual void validate_inputs(const Tensor&                  input,
                                 const Tensor&                  weights,
                                 const hpc::config::ConvConfig& config) const;

    // Device management
    virtual void initialize_device() = 0;
    virtual void cleanup_device()    = 0;
};

} // namespace compute
} // namespace hpc