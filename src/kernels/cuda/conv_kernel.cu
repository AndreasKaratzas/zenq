#include "backend/kernels/conv_kernel.cuh"
#include <cuda_runtime.h>

namespace kernel_compute {

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
                                     int      padding) {
    const int b  = blockIdx.x;
    const int oc = blockIdx.y;
    const int oh = blockIdx.z / ((width - kernel_size + 2 * padding) / stride + 1);
    const int ow = blockIdx.z % ((width - kernel_size + 2 * padding) / stride + 1);

    if (b >= batch_size || oc >= out_channels)
        return;

    T         sum     = 0;
    const int h_start = oh * stride - padding;
    const int w_start = ow * stride - padding;

    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                const int h = h_start + kh;
                const int w = w_start + kw;

                if (h >= 0 && h < height && w >= 0 && w < width) {
                    const int input_idx = ((b * in_channels + ic) * height + h) * width + w;
                    const int weight_idx =
                        ((oc * in_channels + ic) * kernel_size + kh) * kernel_size + kw;
                    sum += input[input_idx] * weights[weight_idx];
                }
            }
        }
    }

    const int out_height = (height - kernel_size + 2 * padding) / stride + 1;
    const int out_width  = (width - kernel_size + 2 * padding) / stride + 1;
    const int output_idx = ((b * out_channels + oc) * out_height + oh) * out_width + ow;
    output[output_idx]   = sum;
}

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
                                      int            padding) {
    const int b  = blockIdx.x;
    const int oc = blockIdx.y;
    const int oh = blockIdx.z / ((width - kernel_size + 2 * padding) / stride + 1);
    const int ow = blockIdx.z % ((width - kernel_size + 2 * padding) / stride + 1);

    if (b >= batch_size || oc >= out_channels)
        return;

    T         sum     = 0;
    const int h_start = oh * stride - padding;
    const int w_start = ow * stride - padding;

    const int row = oc * (in_channels * kernel_size * kernel_size);
    for (int idx = row_ptr[row]; idx < row_ptr[row + 1]; ++idx) {
        const int col       = col_ind[idx];
        const int ic        = col / (kernel_size * kernel_size);
        const int remainder = col % (kernel_size * kernel_size);
        const int kh        = remainder / kernel_size;
        const int kw        = remainder % kernel_size;

        const int h = h_start + kh;
        const int w = w_start + kw;

        if (h >= 0 && h < height && w >= 0 && w < width) {
            const int input_idx = ((b * in_channels + ic) * height + h) * width + w;
            sum += input[input_idx] * weights[idx];
        }
    }

    const int out_height = (height - kernel_size + 2 * padding) / stride + 1;
    const int out_width  = (width - kernel_size + 2 * padding) / stride + 1;
    const int output_idx = ((b * out_channels + oc) * out_height + oh) * out_width + ow;
    output[output_idx]   = sum;
}

template <typename T>
Tensor CudaSparseKernel::forward_impl(const Tensor&       input,
                                      const Tensor&       weights,
                                      const KernelConfig& config) {
    validate_inputs(input, weights, config);

    if (!config.row_ptr || !config.col_ind || !config.nnz) {
        throw std::invalid_argument("Sparse kernel requires CSR format data");
    }

    const auto& input_shape  = input.shape();
    const auto  output_shape = compute_output_shape(input_shape, config);
    Tensor      output(output_shape, input.dtype());

    // Allocate device memory for sparse matrix format
    int32_t*     d_row_ptr;
    int32_t*     d_col_ind;
    const size_t row_ptr_size = (config.channels + 1) * sizeof(int32_t);
    const size_t col_ind_size = config.nnz.value() * sizeof(int32_t);

    cudaMalloc(&d_row_ptr, row_ptr_size);
    cudaMalloc(&d_col_ind, col_ind_size);
    cudaMemcpy(d_row_ptr, config.row_ptr->data(), row_ptr_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_ind, config.col_ind->data(), col_ind_size, cudaMemcpyHostToDevice);

    const int batch_size  = input_shape[0];
    const int in_channels = input_shape[1];
    const int height      = input_shape[2];
    const int width       = input_shape[3];
    const int out_height  = output_shape[2];
    const int out_width   = output_shape[3];

    const dim3 grid(batch_size, config.channels, out_height * out_width);
    const dim3 block(1, 1, 1);

    sparse_forward_kernel<<<grid, block>>>(input.data<T>().data(),
                                           weights.data<T>().data(),
                                           output.data<T>().data(),
                                           d_row_ptr,
                                           d_col_ind,
                                           batch_size,
                                           in_channels,
                                           config.channels,
                                           height,
                                           width,
                                           config.kernel_size,
                                           config.stride,
                                           config.padding);

    cudaDeviceSynchronize();
    if (cudaGetLastError() != cudaSuccess) {
        throw std::runtime_error("CUDA kernel execution failed");
    }

    cudaFree(d_row_ptr);
    cudaFree(d_col_ind);

    return output;
}

Tensor CudaDenseKernel::forward(const Tensor&       input,
                                const Tensor&       weights,
                                const KernelConfig& config) {
    switch (input.dtype()) {
    case DataType::Float32:
        return forward_impl<float32_t>(input, weights, config);
    case DataType::Float64:
        return forward_impl<float64_t>(input, weights, config);
    default:
        throw std::runtime_error("Unsupported data type for CUDA kernel");
    }
}

Tensor CudaSparseKernel::forward(const Tensor&       input,
                                 const Tensor&       weights,
                                 const KernelConfig& config) {
    switch (input.dtype()) {
    case DataType::Float32:
        return forward_impl<float32_t>(input, weights, config);
    case DataType::Float64:
        return forward_impl<float64_t>(input, weights, config);
    default:
        throw std::runtime_error("Unsupported data type for CUDA kernel");
    }
}

// Explicit template instantiations
template __global__ void dense_forward_kernel<float32_t>(
    const float32_t*, const float32_t*, float32_t*, int, int, int, int, int, int, int, int);
template __global__ void dense_forward_kernel<float64_t>(
    const float64_t*, const float64_t*, float64_t*, int, int, int, int, int, int, int, int);
template __global__ void sparse_forward_kernel<float32_t>(const float32_t*,
                                                          const float32_t*,
                                                          float32_t*,
                                                          const int32_t*,
                                                          const int32_t*,
                                                          int,
                                                          int,
                                                          int,
                                                          int,
                                                          int,
                                                          int,
                                                          int,
                                                          int);
template __global__ void sparse_forward_kernel<float64_t>(const float64_t*,
                                                          const float64_t*,
                                                          float64_t*,
                                                          const int32_t*,
                                                          const int32_t*,
                                                          int,
                                                          int,
                                                          int,
                                                          int,
                                                          int,
                                                          int,
                                                          int,
                                                          int);

} // namespace kernel_compute