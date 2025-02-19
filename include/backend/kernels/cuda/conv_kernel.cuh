#pragma once

#include "kernel_base.hpp"
#include <cuda_runtime.h>

namespace kernel_compute {

// CUDA kernel declarations
template <typename T>
__global__ void dense_forward_kernel(const T* input,
                                     const T* weights,
                                     T*       output,
                                     int      batch_size,
                                     int      in_channels,
                                     int      out_channels,
                                     int      height,
                                     int      width,
                                     int      kernel_size,
                                     int      stride,
                                     int      padding);

template <typename T>
__global__ void sparse_forward_kernel(const T*       input,
                                      const T*       weights,
                                      T*             output,
                                      const int32_t* row_ptr,
                                      const int32_t* col_ind,
                                      int            batch_size,
                                      int            in_channels,
                                      int            out_channels,
                                      int            height,
                                      int            width,
                                      int            kernel_size,
                                      int            stride,
                                      int            padding);

class CudaDenseKernel : public KernelBase {
public:
    Tensor forward(const Tensor& input, const Tensor& weights, const KernelConfig& config) override;

private:
    template <typename T>
    Tensor forward_impl(const Tensor& input, const Tensor& weights, const KernelConfig& config);
};

class CudaSparseKernel : public KernelBase {
public:
    Tensor forward(const Tensor& input, const Tensor& weights, const KernelConfig& config) override;

private:
    template <typename T>
    Tensor forward_impl(const Tensor& input, const Tensor& weights, const KernelConfig& config);
};

} // namespace kernel_compute