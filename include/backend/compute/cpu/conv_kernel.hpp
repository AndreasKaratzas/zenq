#pragma once

#include "backend/compute/kernel_base.hpp"

namespace hpc {
namespace compute {
namespace cpu {

class ConvKernel : public KernelBase {
public:
    ConvKernel();
    ~ConvKernel() override;

    Tensor forward(const Tensor&                  input,
                   const Tensor&                  weights,
                   const hpc::config::ConvConfig& config) override;

protected:
    void initialize_device() override;
    void cleanup_device() override;

private:
    template <hpc::core::TensorDataType T>
    void compute_conv2d(const Tensor&                  input,
                        const Tensor&                  weights,
                        Tensor&                        output,
                        const hpc::config::ConvConfig& config);
};

} // namespace cpu
} // namespace compute
} // namespace hpc