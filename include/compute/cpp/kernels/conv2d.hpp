#pragma once

#include "compute/cpp/kernel.hpp"
#include <array> // If you need std::array

namespace hpc::compute {

struct CPUFeatures {
    bool sse4_2;
    bool avx;
    bool fma;
    bool avx2;
    bool avx512f;

    static CPUFeatures detect(); // Declaration only
};

enum class ConvImplementation {
    Basic,
    SSE42,
    AVX2,
    AVX512
};

// Conv2D class *declaration*.  Definition is in conv2d.cpp.
template <typename T>
class Conv2D : public BaseKernel<T> {
public:
    explicit Conv2D(const KernelDescriptor& desc);

    void                           validate_input(const Tensor<T>& input) const override;
    [[nodiscard]] Tensor<T>        forward(const Tensor<T>& input,
                                           bool             return_to_cpu = true) const override;
    void                           load_weights(Tensor<T>&& weights) override;
    [[nodiscard]] const Tensor<T>& weights() const override {
        return weights_;
    }

private:
    void                                    initialize_weights();
    [[nodiscard]] std::pair<size_t, size_t> get_output_dims(const Tensor<T>& input) const;
    Tensor<T>                               weights_;

    // SIMD-related members
    static const CPUFeatures         cpu_features_; // Declaration
    [[nodiscard]] ConvImplementation select_implementation() const;

    [[nodiscard]] Tensor<T> forward_basic(const Tensor<T>& input) const;
    [[nodiscard]] Tensor<T> forward_sse42(const Tensor<T>& input) const;
    [[nodiscard]] Tensor<T> forward_avx2(const Tensor<T>& input) const;
    [[nodiscard]] Tensor<T> forward_avx512(const Tensor<T>& input) const;
};

// Declaration of make_conv2d.
template <typename T>
std::unique_ptr<BaseKernel<T>> make_conv2d(size_t kernel_height,
                                           size_t kernel_width,
                                           size_t in_channels,
                                           size_t out_channels,
                                           size_t stride  = 1,
                                           size_t padding = 0);

} // namespace hpc::compute