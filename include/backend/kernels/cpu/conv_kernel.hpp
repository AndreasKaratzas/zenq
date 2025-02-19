#pragma once

#include "kernel_base.hpp"
#include <immintrin.h>

namespace kernel_compute {

class CpuDenseKernel : public KernelBase {
public:
    Tensor forward(const Tensor& input, const Tensor& weights, const KernelConfig& config) override;

private:
    template <typename T>
    Tensor forward_impl(const Tensor& input, const Tensor& weights, const KernelConfig& config);

    template <typename T>
    void forward_avx512(const T*            input,
                        const T*            weights,
                        T*                  output,
                        const Shape&        input_shape,
                        const KernelConfig& config);
};

class CpuSparseKernel : public KernelBase {
public:
    Tensor forward(const Tensor& input, const Tensor& weights, const KernelConfig& config) override;

private:
    template <typename T>
    Tensor forward_impl(const Tensor& input, const Tensor& weights, const KernelConfig& config);
};

} // namespace kernel_compute