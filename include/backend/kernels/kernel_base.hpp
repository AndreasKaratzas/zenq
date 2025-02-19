#pragma once

#include "common/config.hpp"
#include "tensor.hpp"
#include <memory>

namespace kernel_compute {

class KernelBase {
public:
    virtual ~KernelBase() = default;

    // Main computation interface
    virtual Tensor forward(const Tensor&       input,
                           const Tensor&       weights,
                           const KernelConfig& config) = 0;

    // Factory method
    static std::unique_ptr<KernelBase> create(DeviceType device, KernelType type);

protected:
    // Utility methods for derived classes
    Shape        compute_output_shape(const Shape& input_shape, const KernelConfig& config) const;
    virtual void validate_inputs(const Tensor&       input,
                                 const Tensor&       weights,
                                 const KernelConfig& config) const;
};

} // namespace kernel_compute