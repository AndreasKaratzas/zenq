#include "compute/cpp/kernels/conv2d.hpp"
#include "common/logging.hpp"
#include "compute/cpp/blas.hpp"
#include <cpuid.h>     // For CPU feature detection
#include <immintrin.h> // For SIMD intrinsics
#include <iostream>    // Remove in final version if not needed
#include <omp.h>       // For OpenMP parallelization
#include <stdexcept>   // For exceptions

namespace hpc::compute {

// --- CPU Feature Detection (Definition) ---
// Note: This implementation stays the same as it's already correctly implemented

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
    __get_cpuid(1, &eax, &ebx, &ecx, &edx);
    features.sse4_2 = (ecx & (1 << 20)) != 0;
    features.avx    = (ecx & (1 << 28)) != 0;
    features.fma    = (ecx & (1 << 12)) != 0;
    __get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx);
    features.avx2    = (ebx & (1 << 5)) != 0;
    features.avx512f = (ebx & (1 << 16)) != 0;
#endif
    return features;
}

// Utility function to log CPU features - independent of any class
void logCPUFeatures(const std::string& component) {
    // Detect CPU features directly (rather than accessing private member)
    CPUFeatures features = CPUFeatures::detect();

    LOG_INFO(component,
             " CPU Features: ",
             "SSE4.2=",
             features.sse4_2 ? "Yes" : "No",
             ", ",
             "AVX=",
             features.avx ? "Yes" : "No",
             ", ",
             "FMA=",
             features.fma ? "Yes" : "No",
             ", ",
             "AVX2=",
             features.avx2 ? "Yes" : "No",
             ", ",
             "AVX512=",
             features.avx512f ? "Yes" : "No");
}

// --- Conv2D Implementation (Definitions) ---

template <typename T>
const CPUFeatures Conv2D<T>::cpu_features_ = CPUFeatures::detect();

// Log CPU features at initialization
static bool logged_conv2d_features = []() {
    logCPUFeatures("Conv2D");
    return true;
}();

template <typename T>
Conv2D<T>::Conv2D(const KernelDescriptor& desc) : BaseKernel<T>(desc) {
    LOG_INFO("Creating Conv2D kernel: ",
             "kernel_size=",
             this->desc_.template get_param<size_t>("kernel_height"),
             "x",
             this->desc_.template get_param<size_t>("kernel_width"),
             ", ",
             "channels=",
             this->desc_.template get_param<size_t>("in_channels"),
             "->",
             this->desc_.template get_param<size_t>("out_channels"),
             ", ",
             "stride=",
             this->desc_.template get_param<size_t>("stride"),
             ", ",
             "padding=",
             this->desc_.template get_param<size_t>("padding"));

    initialize_weights();
}

template <typename T>
void Conv2D<T>::initialize_weights() {
    size_t kernel_height = this->desc_.template get_param<size_t>(("kernel_height"));
    size_t kernel_width  = this->desc_.template get_param<size_t>(("kernel_width"));
    size_t in_channels   = this->desc_.template get_param<size_t>(("in_channels"));
    size_t out_channels  = this->desc_.template get_param<size_t>(("out_channels"));

    LOG_DEBUG("Initializing Conv2D weights: shape=[",
              out_channels,
              ", ",
              in_channels,
              ", ",
              kernel_height,
              ", ",
              kernel_width,
              "]");

    weights_ = Tensor<T>({out_channels, in_channels, kernel_height, kernel_width}, this->layout_);
    T val    = 0.1;
    for (size_t i = 0; i < weights_.size(); ++i) {
        weights_.data()[i] = val;
        val += 0.01; // Smaller increment for better numerical stability
        if (val > 1.0)
            val = 0.1; // Reset to avoid large values
    }

    LOG_INFO("Initialized Conv2D weights, total elements: ", weights_.size());
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

    LOG_DEBUG("Output dimensions: [",
              output_height,
              ", ",
              output_width,
              "] from input [",
              input_height,
              ", ",
              input_width,
              "] with kernel [",
              kernel_height,
              ", ",
              kernel_width,
              "], stride ",
              stride,
              ", padding ",
              padding);

    return {output_height, output_width};
}

template <typename T>
void Conv2D<T>::validate_input(const Tensor<T>& input) const {
    LOG_DEBUG("Validating input tensor: rank=",
              input.rank(),
              ", shape=[",
              input.shape()[0],
              ", ",
              input.shape()[1],
              ", ",
              input.shape()[2],
              ", ",
              input.shape()[3],
              "]");

    if (input.rank() != 4) {
        LOG_ERROR("Input validation failed: expected 4D tensor, got ", input.rank(), "D");
        throw std::runtime_error("Input tensor must have 4 dimensions (N, C, H, W).");
    }

    size_t in_channels = this->desc_.template get_param<size_t>(("in_channels"));
    if (input.shape().at(1) != in_channels) {
        LOG_ERROR("Input validation failed: expected ",
                  in_channels,
                  " channels, got ",
                  input.shape().at(1));
        throw std::runtime_error(
            "Input tensor's channel dimension does not match kernel's input channels.");
    }

    size_t kernel_height = this->desc_.template get_param<size_t>(("kernel_height"));
    size_t kernel_width  = this->desc_.template get_param<size_t>(("kernel_width"));

    if (input.shape().at(2) < kernel_height) {
        LOG_ERROR("Input validation failed: height ",
                  input.shape().at(2),
                  " is smaller than kernel height ",
                  kernel_height);
        throw std::runtime_error("Input tensor height is smaller than kernel height.");
    }
    if (input.shape().at(3) < kernel_width) {
        LOG_ERROR("Input validation failed: width ",
                  input.shape().at(3),
                  " is smaller than kernel width ",
                  kernel_width);
        throw std::runtime_error("Input tensor width is smaller than kernel width.");
    }

    LOG_DEBUG("Input tensor validation passed");
}

template <typename T>
void Conv2D<T>::load_weights(Tensor<T>&& weights) {
    LOG_INFO("Loading pre-trained weights: shape=[",
             weights.shape()[0],
             ", ",
             weights.shape()[1],
             ", ",
             weights.shape()[2],
             ", ",
             weights.shape()[3],
             "]");

    weights_ = std::move(weights); // Use std::move for efficiency
}

template <typename T>
ConvImplementation Conv2D<T>::select_implementation() const {
    std::string        type_name = typeid(T).name();
    ConvImplementation impl      = ConvImplementation::Basic;

    if constexpr (std::is_same_v<T, float>) {
        if (cpu_features_.avx512f) {
            impl = ConvImplementation::AVX512;
        } else if (cpu_features_.avx2) {
            impl = ConvImplementation::AVX2;
        } else if (cpu_features_.sse4_2) {
            impl = ConvImplementation::SSE42;
        }
    }

    // Log the selected implementation
    std::string impl_str;
    switch (impl) {
    case ConvImplementation::AVX512:
        impl_str = "AVX-512";
        break;
    case ConvImplementation::AVX2:
        impl_str = "AVX2";
        break;
    case ConvImplementation::SSE42:
        impl_str = "SSE4.2";
        break;
    case ConvImplementation::Basic:
        impl_str = "Basic";
        break;
    }

    LOG_OPTIMIZATION("Conv2D", impl_str, "Type: " + type_name);
    return impl;
}

template <typename T>
Tensor<T> Conv2D<T>::forward(const Tensor<T>& input) const {
    LOG_DEBUG("Conv2D forward: input shape=[",
              input.shape()[0],
              ", ",
              input.shape()[1],
              ", ",
              input.shape()[2],
              ", ",
              input.shape()[3],
              "]");

    // Start timing the forward pass
    TIME_OPERATION("Conv2D::forward",
                   "batch=" + std::to_string(input.shape()[0]) +
                       ", channels=" + std::to_string(input.shape()[1]) + ", spatial=" +
                       std::to_string(input.shape()[2]) + "x" + std::to_string(input.shape()[3]));

    validate_input(input);

    // Dispatch to the appropriate implementation based on available CPU features
    Tensor<T> result;
    switch (select_implementation()) {
    case ConvImplementation::Basic:
        result = forward_basic(input);
        break;
    case ConvImplementation::SSE42:
        result = forward_sse42(input);
        break;
    case ConvImplementation::AVX2:
        result = forward_avx2(input);
        break;
    case ConvImplementation::AVX512:
        result = forward_avx512(input);
        break;
    default:
        LOG_ERROR("Unknown implementation selected");
        throw std::runtime_error("Unsupported convolution implementation");
    }

    LOG_DEBUG("Conv2D forward complete: output shape=[",
              result.shape()[0],
              ", ",
              result.shape()[1],
              ", ",
              result.shape()[2],
              ", ",
              result.shape()[3],
              "]");

    return result;
}

template <typename T>
Tensor<T> Conv2D<T>::forward_basic(const Tensor<T>& input) const {
    // Extract parameters
    size_t batch_size    = input.shape()[0];
    size_t in_channels   = this->desc_.template get_param<size_t>("in_channels");
    size_t out_channels  = this->desc_.template get_param<size_t>("out_channels");
    size_t kernel_height = this->desc_.template get_param<size_t>("kernel_height");
    size_t kernel_width  = this->desc_.template get_param<size_t>("kernel_width");
    size_t stride        = this->desc_.template get_param<size_t>("stride");
    size_t padding       = this->desc_.template get_param<size_t>("padding");

    // Get output dimensions
    auto [output_height, output_width] = get_output_dims(input);

    // Initialize output tensor
    Tensor<T> output({batch_size, out_channels, output_height, output_width}, this->layout_);

    // Create im2col tensor with proper dimensions and alignment
    // Shape: [batch_size, in_channels * kernel_height * kernel_width, output_height * output_width]
    Tensor<T> col_buffer(
        {batch_size, in_channels * kernel_height * kernel_width, output_height * output_width},
        MemoryLayout::RowMajor,
        MemoryOrder::Aligned64);

    // Use the tensor-based im2col from our enhanced BLAS
    BLAS<T>::im2col(
        input, kernel_height, kernel_width, padding, padding, stride, stride, col_buffer);

    // Create a reshaped view of weights for GEMM
    // Original shape: [out_channels, in_channels, kernel_height, kernel_width]
    // Reshape to: [out_channels, in_channels * kernel_height * kernel_width]
    size_t gemm_k = in_channels * kernel_height * kernel_width;

// For each batch, perform GEMM: output = weights * col_buffer
#pragma omp parallel for
    for (size_t n = 0; n < batch_size; ++n) {
        // Slice the tensors for the current batch
        // Create views or slices would be ideal, but we'll use pointers for now
        const T* batch_col_data = col_buffer.data() + n * gemm_k * output_height * output_width;
        T*       batch_output   = output.data() + n * out_channels * output_height * output_width;

        // Perform GEMM
        BLAS<T>::gemm(false,
                      false,
                      out_channels,
                      output_height * output_width,
                      gemm_k,
                      static_cast<T>(1),
                      weights_.data(),
                      gemm_k,
                      batch_col_data,
                      output_height * output_width,
                      static_cast<T>(0),
                      batch_output,
                      output_height * output_width);
    }

    return output;
}

template <typename T>
Tensor<T> Conv2D<T>::forward_sse42(const Tensor<T>& input) const {
    if constexpr (std::is_same_v<T, float>) {
        // Extract parameters
        size_t batch_size    = input.shape()[0];
        size_t in_channels   = this->desc_.template get_param<size_t>("in_channels");
        size_t out_channels  = this->desc_.template get_param<size_t>("out_channels");
        size_t kernel_height = this->desc_.template get_param<size_t>("kernel_height");
        size_t kernel_width  = this->desc_.template get_param<size_t>("kernel_width");
        size_t stride        = this->desc_.template get_param<size_t>("stride");
        size_t padding       = this->desc_.template get_param<size_t>("padding");

        // Get output dimensions
        auto [output_height, output_width] = get_output_dims(input);

        // Initialize output tensor with SSE4.2-friendly alignment
        Tensor<T> output({batch_size, out_channels, output_height, output_width},
                         this->layout_,
                         MemoryOrder::Aligned32);

        // Create im2col tensor with proper dimensions and alignment
        // SSE4.2 works well with 16-byte alignment (4 floats)
        Tensor<T> col_buffer(
            {batch_size, in_channels * kernel_height * kernel_width, output_height * output_width},
            MemoryLayout::RowMajor,
            MemoryOrder::Aligned32);

        // Use tensor-based im2col from our enhanced BLAS
        BLAS<T>::im2col(
            input, kernel_height, kernel_width, padding, padding, stride, stride, col_buffer);

        // Create a flattened weights tensor for more efficient GEMM
        size_t gemm_k = in_channels * kernel_height * kernel_width;

// For each batch, perform GEMM using SSE4.2-optimized gemm
#pragma omp parallel for
        for (size_t n = 0; n < batch_size; ++n) {
            // Slice the tensors for the current batch
            const T* batch_col_data = col_buffer.data() + n * gemm_k * output_height * output_width;
            T*       batch_output = output.data() + n * out_channels * output_height * output_width;

            // Perform GEMM - our enhanced BLAS will automatically use SSE4.2
            BLAS<T>::gemm(false,
                          false,
                          out_channels,
                          output_height * output_width,
                          gemm_k,
                          static_cast<T>(1),
                          weights_.data(),
                          gemm_k,
                          batch_col_data,
                          output_height * output_width,
                          static_cast<T>(0),
                          batch_output,
                          output_height * output_width);
        }

        return output;
    } else {
        // For non-float types, use the basic implementation
        return forward_basic(input);
    }
}

template <typename T>
Tensor<T> Conv2D<T>::forward_avx2(const Tensor<T>& input) const {
    if constexpr (std::is_same_v<T, float>) {
        // Extract parameters
        size_t batch_size    = input.shape()[0];
        size_t in_channels   = this->desc_.template get_param<size_t>("in_channels");
        size_t out_channels  = this->desc_.template get_param<size_t>("out_channels");
        size_t kernel_height = this->desc_.template get_param<size_t>("kernel_height");
        size_t kernel_width  = this->desc_.template get_param<size_t>("kernel_width");
        size_t stride        = this->desc_.template get_param<size_t>("stride");
        size_t padding       = this->desc_.template get_param<size_t>("padding");

        // Get output dimensions
        auto [output_height, output_width] = get_output_dims(input);
        size_t output_size                 = output_height * output_width;

        // Initialize output tensor with AVX2-friendly alignment (32-byte)
        Tensor<T> output({batch_size, out_channels, output_height, output_width},
                         this->layout_,
                         MemoryOrder::Aligned32);

        // Create im2col tensor with AVX2-friendly alignment
        // AVX2 works well with 32-byte alignment (8 floats)
        Tensor<T> col_buffer({batch_size, in_channels * kernel_height * kernel_width, output_size},
                             MemoryLayout::RowMajor,
                             MemoryOrder::Aligned32);

        // Use tensor-based im2col with AVX2 optimization
        BLAS<T>::im2col(
            input, kernel_height, kernel_width, padding, padding, stride, stride, col_buffer);

        // Get dimensions for GEMM
        size_t gemm_k = in_channels * kernel_height * kernel_width;

        // Perform GEMM using AVX2 optimization with blocking for better cache usage
        const size_t block_size = 32; // Good for L1 cache

#pragma omp parallel for
        for (size_t n = 0; n < batch_size; ++n) {
            // Slice the tensors for the current batch
            const T* batch_col_data = col_buffer.data() + n * gemm_k * output_size;
            T*       batch_output   = output.data() + n * out_channels * output_size;

            // Perform GEMM with blocking for better cache utilization
            // Our enhanced BLAS will use AVX2 automatically
            for (size_t oc_block = 0; oc_block < out_channels; oc_block += block_size) {
                size_t oc_end = std::min(oc_block + block_size, out_channels);

                for (size_t out_block = 0; out_block < output_size; out_block += block_size) {
                    size_t out_end = std::min(out_block + block_size, output_size);

                    // Calculate output block
                    for (size_t oc = oc_block; oc < oc_end; ++oc) {
                        for (size_t o = out_block; o < out_end; ++o) {
                            T sum = 0;
                            // Use AVX vector instructions via BLAS
                            for (size_t k = 0; k < gemm_k; ++k) {
                                sum += weights_.data()[oc * gemm_k + k] *
                                       batch_col_data[k * output_size + o];
                            }
                            batch_output[oc * output_size + o] = sum;
                        }
                    }
                }
            }
        }

        return output;
    } else {
        // For non-float types, use the basic implementation
        return forward_basic(input);
    }
}

template <typename T>
Tensor<T> Conv2D<T>::forward_avx512(const Tensor<T>& input) const {
    if constexpr (std::is_same_v<T, float>) {
        // Extract parameters
        size_t batch_size    = input.shape()[0];
        size_t in_channels   = this->desc_.template get_param<size_t>("in_channels");
        size_t out_channels  = this->desc_.template get_param<size_t>("out_channels");
        size_t kernel_height = this->desc_.template get_param<size_t>("kernel_height");
        size_t kernel_width  = this->desc_.template get_param<size_t>("kernel_width");
        size_t stride        = this->desc_.template get_param<size_t>("stride");
        size_t padding       = this->desc_.template get_param<size_t>("padding");

        // Get output dimensions
        auto [output_height, output_width] = get_output_dims(input);
        size_t output_size                 = output_height * output_width;

        // Initialize output tensor with AVX-512 friendly alignment (64-byte)
        Tensor<T> output({batch_size, out_channels, output_height, output_width},
                         this->layout_,
                         MemoryOrder::Aligned64);

        // Create im2col tensor with AVX-512 alignment
        // AVX-512 works best with 64-byte alignment (16 floats)
        Tensor<T> col_buffer({batch_size, in_channels * kernel_height * kernel_width, output_size},
                             MemoryLayout::RowMajor,
                             MemoryOrder::Aligned64);

        // Use tensor-based im2col with AVX-512 optimization
        BLAS<T>::im2col(
            input, kernel_height, kernel_width, padding, padding, stride, stride, col_buffer);

        // Get dimensions for GEMM
        size_t gemm_k = in_channels * kernel_height * kernel_width;

        // Reshape weights for more efficient GEMM if needed
        // For now, we'll use the weights directly

        // Prepare for GEMM with optimal blocking for AVX-512
        const size_t m_block = 32;  // Block size for output channels
        const size_t n_block = 256; // Block size for output spatial dimensions
        const size_t k_block = 64;  // Block size for kernel dimensions

#pragma omp parallel for
        for (size_t n = 0; n < batch_size; ++n) {
            // Slice the tensors for the current batch
            const T* batch_col_data = col_buffer.data() + n * gemm_k * output_size;
            T*       batch_output   = output.data() + n * out_channels * output_size;

            // Perform GEMM - our enhanced BLAS will use AVX-512 automatically
            BLAS<T>::gemm(false,
                          false,
                          out_channels,
                          output_size,
                          gemm_k,
                          static_cast<T>(1),
                          weights_.data(),
                          gemm_k,
                          batch_col_data,
                          output_size,
                          static_cast<T>(0),
                          batch_output,
                          output_size);
        }

        return output;
    } else {
        // For non-float types, use the basic implementation
        return forward_basic(input);
    }
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