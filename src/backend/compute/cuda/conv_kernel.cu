#include "backend/compute/cuda/conv_kernel.cuh"
#include <cuda_runtime.h>

namespace hpc {
namespace compute {
namespace cuda {

// CUDA kernel for direct convolution
template <typename T>
__global__ void conv2d_kernel(const T* __restrict__ input,
                              const T* __restrict__ weights,
                              T* __restrict__ output,
                              const int batch_size,
                              const int in_channels,
                              const int in_height,
                              const int in_width,
                              const int out_channels,
                              const int out_height,
                              const int out_width,
                              const int kernel_size,
                              const int stride,
                              const int padding) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int bz = blockIdx.z;

    // Calculate output position
    const int ow = bx * blockDim.x + tx;
    const int oh = by * blockDim.y + ty;
    const int oc = bz % out_channels;
    const int n  = bz / out_channels;

    if (n >= batch_size || oc >= out_channels || oh >= out_height || ow >= out_width)
        return;

    // Shared memory for input tile and weights
    extern __shared__ float shared_mem[];
    T*                      shared_input = shared_mem;
    T*                      shared_weights =
        shared_mem + (blockDim.x + kernel_size - 1) * (blockDim.y + kernel_size - 1);

    T sum = 0;

    // Compute input boundaries
    const int in_h_start = oh * stride - padding;
    const int in_w_start = ow * stride - padding;

    // Loop over input channels
    for (int ic = 0; ic < in_channels; ++ic) {
        // Load weights into shared memory
        if (tx < kernel_size && ty < kernel_size) {
            const int weight_idx = ((oc * in_channels + ic) * kernel_size + ty) * kernel_size + tx;
            shared_weights[ty * kernel_size + tx] = weights[weight_idx];
        }
        __syncthreads();

        // Load input tile into shared memory
        const int tile_h_start = (by * blockDim.y) * stride - padding;
        const int tile_w_start = (bx * blockDim.x) * stride - padding;

        for (int h = ty; h < blockDim.y + kernel_size - 1; h += blockDim.y) {
            for (int w = tx; w < blockDim.x + kernel_size - 1; w += blockDim.x) {
                const int in_h = tile_h_start + h;
                const int in_w = tile_w_start + w;

                const int shared_idx = h * (blockDim.x + kernel_size - 1) + w;
                const int in_idx = ((n * in_channels + ic) * in_height + in_h) * in_width + in_w;

                if (in_h >= 0 && in_h < in_height && in_w >= 0 && in_w < in_width) {
                    shared_input[shared_idx] = input[in_idx];
                } else {
                    shared_input[shared_idx] = 0;
                }
            }
        }
        __syncthreads();

        // Compute convolution for this input channel
        for (int kh = 0; kh < kernel_size; ++kh) {
            const int in_h = in_h_start + kh;
            if (in_h >= 0 && in_h < in_height) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    const int in_w = in_w_start + kw;
                    if (in_w >= 0 && in_w < in_width) {
                        const int shared_in_idx =
                            (ty * stride + kh) * (blockDim.x + kernel_size - 1) +
                            (tx * stride + kw);
                        const int shared_weight_idx = kh * kernel_size + kw;
                        sum += shared_input[shared_in_idx] * shared_weights[shared_weight_idx];
                    }
                }
            }
        }
        __syncthreads();
    }

    // Write output
    const int out_idx = ((n * out_channels + oc) * out_height + oh) * out_width + ow;
    output[out_idx]   = sum;
}

template <hpc::core::DataType T>
void ConvKernel::compute_conv2d(const Tensor&                  input,
                                const Tensor&                  weights,
                                Tensor&                        output,
                                const hpc::config::ConvConfig& config) {
    const auto& input_shape  = input.shape();
    const auto& output_shape = output.shape();

    size_t batch_size   = input_shape[0];
    size_t in_channels  = input_shape[1];
    size_t in_height    = input_shape[2];
    size_t in_width     = input_shape[3];
    size_t out_channels = output_shape[1];
    size_t out_height   = output_shape[2];
    size_t out_width    = output_shape[3];

    // CUDA kernel configuration
    const int BLOCK_SIZE = 16;
    dim3      block(BLOCK_SIZE, BLOCK_SIZE);
    dim3      grid((out_width + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (out_height + BLOCK_SIZE - 1) / BLOCK_SIZE,
              batch_size * out_channels);

    // Calculate shared memory size
    size_t shared_mem_size =
        ((BLOCK_SIZE + config.kernel_size - 1) * (BLOCK_SIZE + config.kernel_size - 1) +
         config.kernel_size * config.kernel_size) *
        sizeof(T);

    // Launch kernel
    conv2d_kernel<T><<<grid, block, shared_mem_size, compute_stream_>>>(input.data_span<T>(),
                                                                        weights.data_span<T>(),
                                                                        output.data_span<T>(),
                                                                        batch_size,
                                                                        in_channels,
                                                                        in_height,
                                                                        in_width,
                                                                        out_channels,
                                                                        out_height,
                                                                        out_width,
                                                                        config.kernel_size,
                                                                        config.stride,
                                                                        config.padding);
}

} // namespace cuda
} // namespace compute
} // namespace hpc