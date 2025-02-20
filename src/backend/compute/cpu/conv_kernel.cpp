#include "backend/compute/cpu/conv_kernel.hpp"
#include <immintrin.h>
#include <omp.h>

namespace hpc {
namespace compute {
namespace cpu {

ConvKernel::ConvKernel() {
    initialize_device();
}

ConvKernel::~ConvKernel() {
    cleanup_device();
}

void ConvKernel::initialize_device() {
    // No specific initialization needed for CPU
}

void ConvKernel::cleanup_device() {
    // No specific cleanup needed for CPU
}

template <hpc::core::TensorDataType T>
void ConvKernel::compute_conv2d(const Tensor&                  input,
                                const Tensor&                  weights,
                                Tensor&                        output,
                                const hpc::config::ConvConfig& config) {
    auto input_data  = input.data_span<T>();
    auto weight_data = weights.data_span<T>();
    auto output_data = output.data_span<T>();

    const auto& input_shape  = input.shape();
    const auto& output_shape = output.shape();

    size_t batch_size   = input_shape[0];
    size_t in_channels  = input_shape[1];
    size_t in_height    = input_shape[2];
    size_t in_width     = input_shape[3];
    size_t out_channels = output_shape[1];
    size_t out_height   = output_shape[2];
    size_t out_width    = output_shape[3];

    // Constants for AVX-512
    constexpr int vec_size          = 16; // Number of float elements in AVX-512 register
    const size_t  out_width_aligned = (out_width + vec_size - 1) / vec_size * vec_size;

    // Allocate aligned temporary buffer for intermediate results
    float* temp_output = static_cast<float*>(aligned_alloc(64, out_width_aligned * sizeof(float)));

#pragma omp parallel for collapse(3)
    for (size_t n = 0; n < batch_size; ++n) {
        for (size_t oc = 0; oc < out_channels; ++oc) {
            for (size_t oh = 0; oh < out_height; ++oh) {
                // Initialize accumulator registers
                __m512* vec_acc = reinterpret_cast<__m512*>(temp_output);
                for (size_t ow = 0; ow < out_width_aligned; ow += vec_size) {
                    _mm512_store_ps(&temp_output[ow], _mm512_setzero_ps());
                }

                for (size_t ic = 0; ic < in_channels; ++ic) {
                    for (size_t kh = 0; kh < config.kernel_size; ++kh) {
                        for (size_t kw = 0; kw < config.kernel_size; ++kw) {
                            int in_h_base = static_cast<int>(oh * config.stride + kh) -
                                            static_cast<int>(config.padding);

                            if (in_h_base >= 0 && in_h_base < static_cast<int>(in_height)) {
                                size_t in_base_offset =
                                    ((n * in_channels + ic) * in_height + in_h_base) * in_width;
                                size_t weight_idx =
                                    ((oc * in_channels + ic) * config.kernel_size + kh) *
                                        config.kernel_size +
                                    kw;
                                __m512 weight_vec = _mm512_set1_ps(weight_data[weight_idx]);

#pragma omp simd
                                for (size_t ow = 0; ow < out_width; ow += vec_size) {
                                    int in_w_base = static_cast<int>(ow * config.stride) -
                                                    static_cast<int>(config.padding);

                                    // Create mask for valid elements
                                    __mmask16 k_mask = _mm512_cmp_ps_mask(
                                        _mm512_set1_ps(static_cast<float>(in_w_base)),
                                        _mm512_set1_ps(
                                            static_cast<float>(in_width - config.kernel_size)),
                                        _CMP_LT_OS);

                                    // Load input data with masking for boundary conditions
                                    __m512 input_vec = _mm512_maskz_loadu_ps(
                                        k_mask, &input_data[in_base_offset + in_w_base]);

                                    // Multiply and accumulate
                                    __m512 acc = _mm512_load_ps(&temp_output[ow]);
                                    acc        = _mm512_fmadd_ps(input_vec, weight_vec, acc);
                                    _mm512_store_ps(&temp_output[ow], acc);
                                }
                            }
                        }
                    }
                }

                // Store final results
                size_t out_offset = ((n * out_channels + oc) * out_height + oh) * out_width;
                for (size_t ow = 0; ow < out_width; ++ow) {
                    output_data[out_offset + ow] = temp_output[ow];
                }
            }
        }
    }

    free(temp_output);
}

Tensor ConvKernel::forward(const Tensor&                  input,
                           const Tensor&                  weights,
                           const hpc::config::ConvConfig& config) {
    validate_inputs(input, weights, config);
    auto   output_shape = compute_output_shape(input.shape(), config);
    Tensor output(output_shape, input.dtype());

    switch (input.dtype()) {
    case hpc::core::DataType::FLOAT32:
        compute_conv2d<float>(input, weights, output, config);
        break;
    case hpc::core::DataType::FLOAT64:
        compute_conv2d<double>(input, weights, output, config);
        break;
    default:
        throw std::runtime_error("Unsupported data type for convolution");
    }

    return output;
}

} // namespace cpu
} // namespace compute
} // namespace hpc