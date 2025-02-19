#include "backend/kernels/cpu/conv_kernel.hpp"
#include <thread>
#include <vector>

namespace kernel_compute {

template <typename T>
void CpuDenseKernel::forward_avx512(const T*            input,
                                    const T*            weights,
                                    T*                  output,
                                    const Shape&        input_shape,
                                    const KernelConfig& config) {
    const int batch_size  = input_shape[0];
    const int in_channels = input_shape[1];
    const int height      = input_shape[2];
    const int width       = input_shape[3];
    const int kernel_size = config.kernel_size;
    const int out_height  = (height - kernel_size + 2 * config.padding) / config.stride + 1;
    const int out_width   = (width - kernel_size + 2 * config.padding) / config.stride + 1;

#pragma omp parallel for collapse(4)
    for (int b = 0; b < batch_size; ++b) {
        for (int oc = 0; oc < config.channels; ++oc) {
            for (int oh = 0; oh < out_height; ++oh) {
                for (int ow = 0; ow < out_width; ++ow) {
                    const int h_start = oh * config.stride - config.padding;
                    const int w_start = ow * config.stride - config.padding;

                    __m512 sum_vec = _mm512_setzero_ps();

                    for (int ic = 0; ic < in_channels; ++ic) {
                        for (int kh = 0; kh < kernel_size; ++kh) {
                            const int h = h_start + kh;
                            if (h < 0 || h >= height)
                                continue;

                            for (int kw = 0; kw < kernel_size; kw += 16) {
                                const int remaining = std::min(16, kernel_size - kw);
                                __mmask16 k_mask    = (1 << remaining) - 1;

                                __m512 input_vec = _mm512_maskz_loadu_ps(
                                    k_mask,
                                    &input[((b * in_channels + ic) * height + h) * width + w_start +
                                           kw]);
                                __m512 weight_vec = _mm512_maskz_loadu_ps(
                                    k_mask,
                                    &weights[((oc * in_channels + ic) * kernel_size + kh) *
                                                 kernel_size +
                                             kw]);

                                sum_vec = _mm512_fmadd_ps(input_vec, weight_vec, sum_vec);
                            }
                        }
                    }

                    output[((b * config.channels + oc) * out_height + oh) * out_width + ow] =
                        _mm512_reduce_add_ps(sum_vec);
                }
            }
        }
    }
}

template <typename T>
Tensor CpuDenseKernel::forward_impl(const Tensor&       input,
                                    const Tensor&       weights,
                                    const KernelConfig& config) {
    validate_inputs(input, weights, config);

    const auto& input_shape  = input.shape();
    const auto  output_shape = compute_output_shape(input_shape, config);
    Tensor      output(output_shape, input.dtype());

    if (config.cpu.use_avx512 && std::is_same_v<T, float32_t>) {
        forward_avx512(input.data<T>().data(),
                       weights.data<T>().data(),
                       output.data<T>().data(),
                       input_shape,
                       config);
    } else {
        const int batch_size  = input_shape[0];
        const int in_channels = input_shape[1];
        const int height      = input_shape[2];
        const int width       = input_shape[3];
        const int out_height  = output_shape[2];
        const int out_width   = output_shape[3];

#pragma omp parallel for collapse(4) num_threads(config.cpu.num_threads)
        for (int b = 0; b < batch_size; ++b) {
            for (int oc = 0; oc < config.channels; ++oc) {
                for (int oh = 0; oh < out_height; ++oh) {
                    for (int ow = 0; ow < out_width; ++ow) {
                        T         sum     = 0;
                        const int h_start = oh * config.stride - config.padding;
                        const int w_start = ow * config.stride - config.padding;

                        for (int ic = 0; ic < in_channels; ++ic) {
                            for (int kh = 0; kh < config.kernel_size; ++kh) {
                                for (int kw = 0; kw < config.kernel_size; ++kw) {
                                    const int h = h_start + kh;
                                    const int w = w_start + kw;

                                    if (h >= 0 && h < height && w >= 0 && w < width) {
                                        const int input_idx =
                                            ((b * in_channels + ic) * height + h) * width + w;
                                        const int weight_idx =
                                            ((oc * in_channels + ic) * config.kernel_size + kh) *
                                                config.kernel_size +
                                            kw;
                                        sum += input.data<T>()[input_idx] *
                                               weights.data<T>()[weight_idx];
                                    }
                                }
                            }
                        }

                        const int output_idx =
                            ((b * config.channels + oc) * out_height + oh) * out_width + ow;
                        output.data<T>()[output_idx] = sum;
                    }
                }
            }
        }
    }

    return output;
}

template <typename T>
Tensor CpuSparseKernel::forward_impl(const Tensor&       input,
                                     const Tensor&       weights,
                                     const KernelConfig& config) {
    validate_inputs(input, weights, config);

    if (!config.row_ptr || !config.col_ind || !config.nnz) {
        throw std::invalid_argument("Sparse kernel requires CSR format data");
    }

    const auto& input_shape  = input.shape();
    const auto  output_shape = compute_output_shape(input_shape, config);
    Tensor      output(output_shape, input.dtype());

    const int batch_size  = input_shape[0];
    const int in_channels = input_shape[1];
    const int height      = input_shape[2];
    const int width       = input_shape[3];
    const int out_height  = output_shape[2];
    const int out_width   = output_shape[3];

#pragma omp parallel for collapse(4) num_threads(config.cpu.num_threads)
    for (int b = 0; b < batch_size; ++b) {
        for (int oc = 0; oc < config.channels; ++oc) {
            for (int oh = 0; oh < out_height; ++oh) {
                for (int ow = 0; ow < out_width; ++ow) {
                    T         sum     = 0;
                    const int h_start = oh * config.stride - config.padding;
                    const int w_start = ow * config.stride - config.padding;

                    const int row = oc * (in_channels * config.kernel_size * config.kernel_size);
                    for (int idx = config.row_ptr->at(row); idx < config.row_ptr->at(row + 1);
                         ++idx) {
                        const int col       = config.col_ind->at(idx);
                        const int ic        = col / (config.kernel_size * config.kernel_size);
                        const int remainder = col % (config.kernel_size * config.kernel_size);
                        const int kh        = remainder / config.kernel_size;
                        const int kw        = remainder % config.kernel_size;

                        const int h = h_start + kh;
                        const int w = w_start + kw;

                        if (h >= 0 && h < height && w >= 0 && w < width) {
                            const int input_idx = ((b * in_channels + ic) * height + h) * width + w;
                            sum += input.data<T>()[input_idx] * weights.data<T>()[idx];
                        }
                    }

                    const int output_idx =
                        ((b * config.channels + oc) * out_height + oh) * out_width + ow;
                    output.data<T>()[output_idx] = sum;
                }
            }
        }
    }

    return output;
}

Tensor CpuDenseKernel::forward(const Tensor&       input,
                               const Tensor&       weights,
                               const KernelConfig& config) {
    switch (input.dtype()) {
    case DataType::Float32:
        return forward_impl<float32_t>(input, weights, config);
    case DataType::Float64:
        return forward_impl<float64_t>(input, weights, config);
    case DataType::Int32:
        return forward_impl<int32_t>(input, weights, config);
    case DataType::Int64:
        return forward_impl<int64_t>(input, weights, config);
    default:
        throw std::runtime_error("Unsupported data type for CPU kernel");
    }
}

Tensor CpuSparseKernel::forward(const Tensor&       input,
                                const Tensor&       weights,
                                const KernelConfig& config) {
    switch (input.dtype()) {
    case DataType::Float32:
        return forward_impl<float32_t>(input, weights, config);
    case DataType::Float64:
        return forward_impl<float64_t>(input, weights, config);
    case DataType::Int32:
        return forward_impl<int32_t>(input, weights, config);
    case DataType::Int64:
        return forward_impl<int64_t>(input, weights, config);
    default:
        throw std::runtime_error("Unsupported data type for CPU kernel");
    }
}

} // namespace kernel_compute