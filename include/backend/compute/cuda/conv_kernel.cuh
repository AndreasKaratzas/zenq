#pragma once

#include "backend/common/types.hpp"
#include "backend/compute/kernel_base.hpp"
#include "backend/compute/tensor.hpp"
#include <cuda_runtime.h>

namespace hpc {
namespace compute {
namespace cuda {

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
    template <hpc::core::DataType T>
    void compute_conv2d(const Tensor&                  input,
                        const Tensor&                  weights,
                        Tensor&                        output,
                        const hpc::config::ConvConfig& config);

    // CUDA specific members
    cudaStream_t compute_stream_;
    void*        workspace_;
    size_t       workspace_size_;
};

} // namespace cuda
} // namespace compute
} // namespace hpc