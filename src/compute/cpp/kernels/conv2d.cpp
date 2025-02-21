#include "compute/cpp/kernels/conv2d.hpp"
#include <cpuid.h>     // For CPU feature detection
#include <immintrin.h> // For SIMD intrinsics
#include <iostream>    // Remove in final version if not needed
#include <stdexcept>   // For exceptions

namespace hpc::compute {

// --- CPU Feature Detection (Definition) ---

CPUFeatures CPUFeatures::detect() {
    CPUFeatures features;
#ifdef _MSC_VER
    int cpu_info[4];
    __cpuid(cpu_info, 1);
    features.sse4_2 = (cpu_info[2] & (1 << 20)) != 0;
    features.avx    = (cpu_info[2] & (1 << 28)) != 0;
    features.fma    = (cpu_info[2] & (1 << 12)) != 0;
    __cpuidex(cpu_info, 7, 0);
    features.avx2    = (cpu_info[1] & (1 << 5)) != 0;
    features.avx512f = (cpu_info[1] & (1 << 16)) != 0;
#else
    unsigned int eax, ebx, ecx, edx;
    __cpuid(1, eax, ebx, ecx, edx);
    features.sse4_2 = (ecx & (1 << 20)) != 0;
    features.avx    = (ecx & (1 << 28)) != 0;
    features.fma    = (ecx & (1 << 12)) != 0;
    __cpuid_count(7, 0, eax, ebx, ecx, edx);
    features.avx2    = (ebx & (1 << 5)) != 0;
    features.avx512f = (ebx & (1 << 16)) != 0;
#endif
    return features;
}

// --- Conv2D Implementation (Definitions) ---

template <typename T>
const CPUFeatures Conv2D<T>::cpu_features_ = CPUFeatures::detect();

template <typename T>
Conv2D<T>::Conv2D(const KernelDescriptor& desc) : BaseKernel<T>(desc) {
    initialize_weights();
}

template <typename T>
void Conv2D<T>::initialize_weights() {
    size_t kernel_height = this->desc_.template get_param<size_t>(("kernel_height"));
    size_t kernel_width  = this->desc_.template get_param<size_t>(("kernel_width"));
    size_t in_channels   = this->desc_.template get_param<size_t>(("in_channels"));
    size_t out_channels  = this->desc_.template get_param<size_t>(("out_channels"));

    weights_ = Tensor<T>({out_channels, in_channels, kernel_height, kernel_width}, this->layout_);
    T val    = 0.1;
    for (size_t i = 0; i < weights_.size(); ++i) {
        weights_.data()[i] = val;
        val += 0.1;
    }
}

template <typename T>
std::pair<size_t, size_t> Conv2D<T>::get_output_dims(const Tensor<T>& input) const {
    size_t kernel_height = this->desc_.template get_param<size_t>(("kernel_height"));
    size_t kernel_width  = this->desc_.template get_param<size_t>(("kernel_width"));
    size_t stride        = this->desc_.template get_param<size_t>(("stride"));
    size_t padding       = this->desc_.template get_param<size_t>(("padding"));

    size_t input_height = input.shape().at(2); // Use .at() for bounds checking
    size_t input_width  = input.shape().at(3);

    size_t output_height = (input_height - kernel_height + 2 * padding) / stride + 1;
    size_t output_width  = (input_width - kernel_width + 2 * padding) / stride + 1;

    return {output_height, output_width};
}

template <typename T>
void Conv2D<T>::validate_input(const Tensor<T>& input) const {
    if (input.rank() != 4) { // Use rank() for clarity
        throw std::runtime_error("Input tensor must have 4 dimensions (N, C, H, W).");
    }

    size_t in_channels = this->desc_.template get_param<size_t>(("in_channels"));
    if (input.shape().at(1) != in_channels) {
        throw std::runtime_error(
            "Input tensor's channel dimension does not match kernel's input channels.");
    }
    size_t kernel_height = this->desc_.template get_param<size_t>(("kernel_height"));
    size_t kernel_width  = this->desc_.template get_param<size_t>(("kernel_width"));

    if (input.shape().at(2) < kernel_height) {
        throw std::runtime_error("Input tensor height is smaller than kernel height.");
    }
    if (input.shape().at(3) < kernel_width) {
        throw std::runtime_error("Input tensor width is smaller than kernel width.");
    }
}

template <typename T>
void Conv2D<T>::load_weights(Tensor<T>&& weights) {
    weights_ = std::move(weights); // Use std::move for efficiency
}

template <typename T>
ConvImplementation Conv2D<T>::select_implementation() const {
    if constexpr (std::is_same_v<T, float>) { // Only check SIMD for float
        if (cpu_features_.avx512f) {
            return ConvImplementation::AVX512;
        } else if (cpu_features_.avx2 && cpu_features_.fma) {
            return ConvImplementation::AVX2;
        } else if (cpu_features_.sse4_2) {
            return ConvImplementation::SSE42;
        }
    }
    return ConvImplementation::Basic;
}

template <typename T>
Tensor<T> Conv2D<T>::forward(const Tensor<T>& input) const {
    validate_input(input);

    switch (select_implementation()) {
    case ConvImplementation::Basic:
        return forward_basic(input);
    case ConvImplementation::SSE42:
        return forward_sse42(input);
    case ConvImplementation::AVX2:
        return forward_avx2(input);
    case ConvImplementation::AVX512:
        return forward_avx512(input);
    default:
        throw std::runtime_error("Unsupported convolution implementation");
    }
}

template <typename T>
Tensor<T> Conv2D<T>::forward_basic(const Tensor<T>& input) const {
    size_t batch_size    = input.shape()[0];
    size_t in_channels   = this->desc_.template get_param<size_t>(("in_channels"));
    size_t out_channels  = this->desc_.template get_param<size_t>(("out_channels"));
    size_t kernel_height = this->desc_.template get_param<size_t>(("kernel_height"));
    size_t kernel_width  = this->desc_.template get_param<size_t>(("kernel_width"));
    size_t stride        = this->desc_.template get_param<size_t>(("stride"));
    size_t padding       = this->desc_.template get_param<size_t>(("padding"));

    auto [output_height, output_width] = get_output_dims(input);

    Tensor<T> output({batch_size, out_channels, output_height, output_width}, this->layout_);

    // Add padding to the input
    Tensor<T> padded_input({batch_size,
                            in_channels,
                            input.shape().at(2) + 2 * padding,
                            input.shape().at(3) + 2 * padding},
                           this->layout_);

    // Initialize padded_input to 0
    std::fill(padded_input.data(), padded_input.data() + padded_input.size(), static_cast<T>(0));

    // Copy the original input into the padded input
    for (size_t n = 0; n < batch_size; ++n) {
        for (size_t c = 0; c < in_channels; ++c) {
            for (size_t h = 0; h < input.shape().at(2); ++h) {
                for (size_t w = 0; w < input.shape().at(3); ++w) {
                    std::cout << "[Padding Loop] input(n=" << n << ", c=" << c << ", h=" << h
                              << ", w=" << w << ") -> padded_input(n=" << n << ", c=" << c
                              << ", h=" << h + padding << ", w=" << w + padding << ")" << std::endl;
                    padded_input(n, c, h + padding, w + padding) = input(n, c, h, w);
                }
            }
        }
    }

    std::cout << "--- Padding Debug Output ---" << std::endl;
    std::cout << "Input Dimensions: Batch=" << batch_size << ", Channels=" << in_channels
              << ", Height=" << input.shape().at(2) << ", Width=" << input.shape().at(3)
              << std::endl;
    std::cout << "Kernel Dimensions: Height=" << kernel_height << ", Width=" << kernel_width
              << ", In Channels=" << in_channels << ", Out Channels=" << out_channels << std::endl;
    std::cout << "Padding: " << padding << ", Stride: " << stride << std::endl;
    std::cout << "Padded Input Dimensions: Batch=" << padded_input.shape()[0]
              << ", Channels=" << padded_input.shape()[1] << ", Height=" << padded_input.shape()[2]
              << ", Width=" << padded_input.shape()[3] << std::endl;

    std::cout << "\n--- Padded Input (Batch 0, Channel 0, Slice 0-4, 0-4) ---" << std::endl;
    for (size_t h = 0; h < 5; ++h) {
        for (size_t w = 0; w < 5; ++w) {
            std::cout << padded_input(0, 0, h, w) << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "\n--- Weights (Output Channel 0, Input Channel 0) ---" << std::endl;
    for (size_t kh = 0; kh < kernel_height; ++kh) {
        for (size_t kw = 0; kw < kernel_width; ++kw) {
            std::cout << weights_(0, 0, kh, kw) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "--- Convolution Output ---" << std::endl;

    // Perform convolution
    for (size_t n = 0; n < batch_size; ++n) {
        for (size_t oc = 0; oc < out_channels; ++oc) {
            for (size_t oh = 0; oh < output_height; ++oh) {
                for (size_t ow = 0; ow < output_width; ++ow) {
                    T sum = 0;
                    for (size_t ic = 0; ic < in_channels; ++ic) {
                        for (size_t kh = 0; kh < kernel_height; ++kh) {
                            for (size_t kw = 0; kw < kernel_width; ++kw) {
                                T input_val =
                                    padded_input(n, ic, oh * stride + kh, ow * stride + kw);
                                T weight_val = weights_(oc, ic, kh, kw);
                                sum += input_val * weight_val;
                                if (oh == 1 && ow == 1 && oc == 0 && n == 0 &&
                                    ic == 0) { // Debugging center pixel (approx)
                                    std::cout << "  [Center Debug] input_val=" << input_val
                                              << ", weight_val=" << weight_val
                                              << ", partial_sum=" << sum << std::endl;
                                }
                                if (oh == 0 && ow == 0 && oc == 0 && n == 0 &&
                                    ic == 0) { // Debugging corner pixel
                                    std::cout << "  [Corner Debug] input_val=" << input_val
                                              << ", weight_val=" << weight_val
                                              << ", partial_sum=" << sum << std::endl;
                                }
                            }
                        }
                    }
                    output(n, oc, oh, ow) = sum;
                    if (oh == 1 && ow == 1 && oc == 0 && n == 0) {
                        std::cout << "Center Output Value (0,0,1,1): " << sum << std::endl;
                    }
                    if (oh == 0 && ow == 0 && oc == 0 && n == 0) {
                        std::cout << "Corner Output Value (0,0,0,0): " << sum << std::endl;
                    }
                }
            }
        }
    }

    return output;
}

template <typename T>
Tensor<T> Conv2D<T>::forward_sse42(const Tensor<T>& input) const {
    // Placeholder - Replace with actual SSE4.2 implementation
    return forward_basic(input);
}

template <typename T>
Tensor<T> Conv2D<T>::forward_avx2(const Tensor<T>& input) const {
    // Placeholder - Replace with actual AVX2 implementation
    return forward_basic(input);
}

template <typename T>
Tensor<T> Conv2D<T>::forward_avx512(const Tensor<T>& input) const {
    // Placeholder - Replace with actual AVX512 implementation
    return forward_basic(input);
}

// --- make_conv2d (Definition and Instantiation) ---

template <typename T>
std::unique_ptr<BaseKernel<T>> make_conv2d(size_t kernel_height,
                                           size_t kernel_width,
                                           size_t in_channels,
                                           size_t out_channels,
                                           size_t stride,
                                           size_t padding) {
    KernelDescriptor desc(KernelType::Convolution2D);
    desc.set_param("kernel_height", kernel_height);
    desc.set_param("kernel_width", kernel_width);
    desc.set_param("in_channels", in_channels);
    desc.set_param("out_channels", out_channels);
    desc.set_param("stride", stride);
    desc.set_param("padding", padding);
    return std::make_unique<Conv2D<T>>(desc);
}

// Explicit template instantiations
template class Conv2D<float>;
template class Conv2D<double>;

template std::unique_ptr<BaseKernel<float>> make_conv2d<float>(size_t kernel_height,
                                                               size_t kernel_width,
                                                               size_t in_channels,
                                                               size_t out_channels,
                                                               size_t stride,
                                                               size_t padding);

template std::unique_ptr<BaseKernel<double>> make_conv2d<double>(size_t kernel_height,
                                                                 size_t kernel_width,
                                                                 size_t in_channels,
                                                                 size_t out_channels,
                                                                 size_t stride,
                                                                 size_t padding);

} // namespace hpc::compute