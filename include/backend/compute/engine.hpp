#pragma once

#include "common/config.hpp"
#include "kernels/kernel_base.hpp"
#include <memory>

namespace kernel_compute {

class Engine {
public:
    virtual ~Engine() = default;

    // Initialize engine with config
    virtual void initialize(const ComputeConfig& config) = 0;

    // Main computation method
    virtual Tensor compute(const Tensor&       input,
                           const Tensor&       weights,
                           const KernelConfig& kernel_config) = 0;

    // Resource management
    virtual void synchronize() = 0;
    virtual void reset()       = 0;

    // Factory method
    static std::unique_ptr<Engine> create(DeviceType device);

protected:
    std::unique_ptr<KernelBase> kernel_;
    ComputeConfig               config_;
    bool                        initialized_{false};
};

} // namespace kernel_compute