#include "compute/cuda/blas.cuh"
#include "compute/cuda/kernels/conv2d.cuh"
#include <iostream>
#include <stdexcept>

namespace hpc::compute::cuda {

// Implementation of CUDAFeatures detection
CUDAFeatures CUDAFeatures::detect(int deviceId) {
    CUDAFeatures features;

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceId);

    features.computeCapabilityMajor = deviceProp.major;
    features.computeCapabilityMinor = deviceProp.minor;
    features.multiProcessorCount    = deviceProp.multiProcessorCount;
    features.maxThreadsPerBlock     = deviceProp.maxThreadsPerBlock;
    features.totalGlobalMem         = deviceProp.totalGlobalMem;

    return features;
}

// Static member initialization
template <typename T>
const CUDAFeatures Conv2D<T>::cuda_features_ = CUDAFeatures::detect();

// Constructor
template <typename T>
Conv2D<T>::Conv2D(const KernelDescriptor& desc) : BaseKernel<T>(desc) {
    // Create CUDA stream
    cudaStreamCreate(&stream_);

    // Initialize weights
    initialize_weights();

    std::cout << "Created CUDA Conv2D with compute capability "
              << cuda_features_.computeCapabilityMajor << "."
              << cuda_features_.computeCapabilityMinor << std::endl;
}

// Destructor
template <typename T>
Conv2D<T>::~Conv2D() {
    if (stream_ != nullptr) {
        cudaStreamDestroy(stream_);
    }
}

// Initialize weights
template <typename T>
void Conv2D<T>::initialize_weights() {
    size_t kernel_height = this->desc_.template get_param<size_t>("kernel_height");
    size_t kernel_width  = this->desc_.template get_param<size_t>("kernel_width");
    size_t in_channels   = this->desc_.template get_param<size_t>("in_channels");
    size_t out_channels  = this->desc_.template get_param<size_t>("out_channels");

    // First create on CPU
    Tensor<T> cpu_weights({out_channels, in_channels, kernel_height, kernel_width});

    // Initialize with some values
    T val = 0.1;
    for (size_t i = 0; i < cpu_weights.size(); ++i) {
        cpu_weights.data()[i] = val;
        val += 0.01;
        if (val > 1.0)
            val = 0.1;
    }

    // Store the host copy
    weights_host_ = cpu_weights;

    // Transfer to GPU
    weights_ = TensorWrapper<T>(cpu_weights);
}

// Validate input tensor
template <typename T>
void Conv2D<T>::validate_input(const Tensor<T>& input) const {
    // Convert to TensorWrapper
    TensorWrapper<T> input_wrapper(input);

    if (input_wrapper.rank() != 4) {
        throw std::runtime_error("Input tensor must have 4 dimensions (N, C, H, W).");
    }

    size_t in_channels = this->desc_.template get_param<size_t>("in_channels");
    if (input_wrapper.dims()[1] != in_channels) {
        throw std::runtime_error(
            "Input tensor's channel dimension does not match kernel's input channels.");
    }

    size_t kernel_height = this->desc_.template get_param<size_t>("kernel_height");
    size_t kernel_width  = this->desc_.template get_param<size_t>("kernel_width");

    if (input_wrapper.dims()[2] < kernel_height) {
        throw std::runtime_error("Input tensor height is smaller than kernel height.");
    }
    if (input_wrapper.dims()[3] < kernel_width) {
        throw std::runtime_error("Input tensor width is smaller than kernel width.");
    }
}

// Calculate output dimensions
template <typename T>
std::pair<size_t, size_t> Conv2D<T>::get_output_dims(const TensorWrapper<T>& input) const {
    size_t kernel_height = this->desc_.template get_param<size_t>("kernel_height");
    size_t kernel_width  = this->desc_.template get_param<size_t>("kernel_width");
    size_t stride        = this->desc_.template get_param<size_t>("stride");
    size_t padding       = this->desc_.template get_param<size_t>("padding");

    size_t input_height = input.dims()[2];
    size_t input_width  = input.dims()[3];

    size_t output_height = (input_height - kernel_height + 2 * padding) / stride + 1;
    size_t output_width  = (input_width - kernel_width + 2 * padding) / stride + 1;

    return {output_height, output_width};
}

// Select implementation based on hardware capabilities
template <typename T>
ConvImplementation Conv2D<T>::select_implementation() const {
    // Basic strategy for selecting implementation
    if (cuda_features_.computeCapabilityMajor >= 7) {
        // For Volta and above, cuBLAS with Tensor Cores is fastest
        return ConvImplementation::CuBLAS;
    } else if (cuda_features_.computeCapabilityMajor >= 6) {
        // For Pascal, Winograd can be faster for 3x3 kernels
        size_t kernel_height = this->desc_.template get_param<size_t>("kernel_height");
        size_t kernel_width  = this->desc_.template get_param<size_t>("kernel_width");

        if (kernel_height == 3 && kernel_width == 3) {
            return ConvImplementation::WinogradFast;
        } else {
            return ConvImplementation::CuBLAS;
        }
    } else {
        // For older GPUs, tiled implementation works well
        return ConvImplementation::Tiled;
    }
}

// Standard forward method
template <typename T>
Tensor<T> Conv2D<T>::forward(const Tensor<T>& input, bool return_to_cpu) const {
    // Validate input
    this->validate_input(input);

    // Convert input Tensor to TensorWrapper
    TensorWrapper<T> input_wrapper(input);

    // Use forward_gpu to perform computation
    TensorWrapper<T> result = forward_gpu(input_wrapper);

    // If requested, convert result back to CPU
    if (return_to_cpu) {
        // Get output dimensions from the input and kernel parameters
        auto                out_dims  = get_output_dims(input_wrapper);
        std::vector<size_t> out_shape = {
            input.shape()[0],           // batch size
            this->weights().shape()[0], // out_channels (first dim of weights tensor)
            out_dims.first,             // height
            out_dims.second             // width
        };

        // Allocate output tensor with the right dimensions
        Tensor<T> output_tensor(out_shape);

        // Pass the entire tensor to copy_to_host
        result.copy_to_host(output_tensor);

        return output_tensor;
    } else {
        // Return an empty tensor if we're not copying back to CPU
        return Tensor<T>();
    }
}

// GPU-to-GPU forward method
template <typename T>
TensorWrapper<T> Conv2D<T>::forward_gpu(const TensorWrapper<T>& input) const {
    // Dispatch to appropriate implementation
    switch (select_implementation()) {
    case ConvImplementation::Basic:
        return forward_basic(input);
    case ConvImplementation::Tiled:
        return forward_tiled(input);
    case ConvImplementation::CuBLAS:
        return forward_cublas(input);
    case ConvImplementation::WinogradFast:
        return forward_winograd(input);
    default:
        throw std::runtime_error("Unknown implementation selected");
    }
}

// Utility method to get output shape for a given input
template <typename T>
std::vector<size_t> Conv2D<T>::get_output_shape(const Tensor<T>& input) const {
    // Create a temporary wrapper to use get_output_dims
    TensorWrapper<T> temp_wrapper(input);
    auto             out_dims = get_output_dims(temp_wrapper);

    return {
        input.shape()[0],           // batch
        this->weights().shape()[0], // out_channels (first dim of weights tensor)
        out_dims.first,             // height
        out_dims.second             // width
    };
}

// Load pre-trained weights
template <typename T>
void Conv2D<T>::load_weights(Tensor<T>&& weights) {
    // Store host copy
    weights_host_ = std::move(weights);

    // Create device copy
    weights_ = TensorWrapper<T>(weights_host_);
}

// CUDA kernel for im2col operation
template <typename T>
__global__ void im2col_kernel(const T* data_im,
                              int      channels,
                              int      height,
                              int      width,
                              int      kernel_h,
                              int      kernel_w,
                              int      pad_h,
                              int      pad_w,
                              int      stride_h,
                              int      stride_w,
                              int      output_h,
                              int      output_w,
                              T*       data_col) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < channels * kernel_h * kernel_w * output_h * output_w) {
        int w_out = index % output_w;
        index /= output_w;
        int h_out = index % output_h;
        index /= output_h;
        int w_in = index % kernel_w;
        index /= kernel_w;
        int h_in = index % kernel_h;
        index /= kernel_h;
        int c_in = index;

        int h_im = h_out * stride_h - pad_h + h_in;
        int w_im = w_out * stride_w - pad_w + w_in;

        data_col[(c_in * kernel_h * kernel_w + h_in * kernel_w + w_in) * output_h * output_w +
                 h_out * output_w + w_out] =
            (h_im >= 0 && h_im < height && w_im >= 0 && w_im < width)
                ? data_im[(c_in * height + h_im) * width + w_im]
                : T(0);
    }
}

// Basic forward implementation using im2col + GEMM
template <typename T>
TensorWrapper<T> Conv2D<T>::forward_basic(const TensorWrapper<T>& input) const {
    // Extract parameters
    size_t batch_size    = input.dims()[0];
    size_t in_channels   = this->desc_.template get_param<size_t>("in_channels");
    size_t out_channels  = this->desc_.template get_param<size_t>("out_channels");
    size_t kernel_height = this->desc_.template get_param<size_t>("kernel_height");
    size_t kernel_width  = this->desc_.template get_param<size_t>("kernel_width");
    size_t stride        = this->desc_.template get_param<size_t>("stride");
    size_t padding       = this->desc_.template get_param<size_t>("padding");

    // Get output dimensions
    auto [output_height, output_width] = get_output_dims(input);

    // Initialize output tensor
    TensorWrapper<T> output({batch_size, out_channels, output_height, output_width});

    // Create im2col tensor
    size_t           gemm_k = in_channels * kernel_height * kernel_width;
    TensorWrapper<T> col_buffer({batch_size, gemm_k, output_height * output_width});

    // Calculate total threads for kernel launch
    const int threads_per_block = 256;
    const int total_elements =
        in_channels * kernel_height * kernel_width * output_height * output_width;
    const int blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    // For each batch
    for (size_t n = 0; n < batch_size; ++n) {
        // Perform im2col
        const T* input_data =
            input.tensor_data().data + n * in_channels * input.dims()[2] * input.dims()[3];
        T* col_data = col_buffer.tensor_data().data + n * gemm_k * output_height * output_width;

        // Launch im2col kernel
        im2col_kernel<<<blocks, threads_per_block, 0, stream_>>>(input_data,
                                                                 in_channels,
                                                                 input.dims()[2],
                                                                 input.dims()[3],
                                                                 kernel_height,
                                                                 kernel_width,
                                                                 padding,
                                                                 padding,
                                                                 stride,
                                                                 stride,
                                                                 output_height,
                                                                 output_width,
                                                                 col_data);

        // Perform GEMM: output = weights * col_buffer
        T* output_data =
            output.tensor_data().data + n * out_channels * output_height * output_width;

        // Use BLAS with gemm (not gemm_custom since it's likely private)
        BLAS<T>::gemm(false,
                      false,
                      out_channels,
                      output_height * output_width,
                      gemm_k,
                      T(1.0),
                      weights_.tensor_data().data,
                      gemm_k,
                      col_data,
                      output_height * output_width,
                      T(0.0),
                      output_data,
                      output_height * output_width,
                      stream_);
    }

    // Return the result
    return output;
}

template <typename T>
__global__ void conv2d_tiled_kernel(const T* input,
                                    const T* kernel,
                                    T*       output,
                                    int      batch_size,
                                    int      in_channels,
                                    int      out_channels,
                                    int      height,
                                    int      width,
                                    int      kernel_h,
                                    int      kernel_w,
                                    int      pad_h,
                                    int      pad_w,
                                    int      stride_h,
                                    int      stride_w,
                                    int      output_h,
                                    int      output_w) {
    // Shared memory for input tile and kernel
    __shared__ T shared_input[TILE_SIZE + 2][TILE_SIZE + 2]; // +2 for padding assuming 3x3 kernel
    __shared__ T shared_kernel[3][3];                        // Assuming 3x3 kernel for simplicity

    // Block and thread indices
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Calculate output coordinates
    int batch_idx   = blockIdx.z / out_channels;
    int out_channel = blockIdx.z % out_channels;
    int out_row     = by * BLOCK_SIZE + ty;
    int out_col     = bx * BLOCK_SIZE + tx;

    // Check if this thread computes an output element
    bool compute_output = (out_row < output_h && out_col < output_w);

    // Initialize accumulator
    T result = 0;

    // Loop over input channels
    for (int in_c = 0; in_c < in_channels; in_c++) {
        // Load kernel weights into shared memory
        if (ty < 3 && tx < 3) {
            shared_kernel[ty][tx] =
                kernel[((out_channel * in_channels + in_c) * kernel_h + ty) * kernel_w + tx];
        }
        __syncthreads();

        // Calculate input position for this output element
        int in_row_base = out_row * stride_h - pad_h;
        int in_col_base = out_col * stride_w - pad_w;

        // Collaboratively load input tile into shared memory
        for (int i = ty; i < TILE_SIZE + 2; i += BLOCK_SIZE) {
            for (int j = tx; j < TILE_SIZE + 2; j += BLOCK_SIZE) {
                int in_row = in_row_base - 1 + i; // -1 for padding offset
                int in_col = in_col_base - 1 + j; // -1 for padding offset

                // Check bounds and load data or zero
                if (in_row >= 0 && in_row < height && in_col >= 0 && in_col < width) {
                    shared_input[i][j] =
                        input[((batch_idx * in_channels + in_c) * height + in_row) * width +
                              in_col];
                } else {
                    shared_input[i][j] = 0;
                }
            }
        }
        __syncthreads();

        // Perform convolution using shared memory
        if (compute_output) {
            for (int kh = 0; kh < 3; kh++) {
                for (int kw = 0; kw < 3; kw++) {
                    // Offset for input in shared memory (1 for padding offset)
                    int in_row_offset = ty * stride_h + kh;
                    int in_col_offset = tx * stride_w + kw;

                    result += shared_input[in_row_offset][in_col_offset] * shared_kernel[kh][kw];
                }
            }
        }
        __syncthreads();
    }

    // Write result to global memory
    if (compute_output) {
        output[((batch_idx * out_channels + out_channel) * output_h + out_row) * output_w +
               out_col] = result;
    }
}

template <typename T>
TensorWrapper<T> Conv2D<T>::forward_tiled(const TensorWrapper<T>& input) const {
    // Extract parameters
    size_t batch_size    = input.dims()[0];
    size_t in_channels   = this->desc_.template get_param<size_t>("in_channels");
    size_t out_channels  = this->desc_.template get_param<size_t>("out_channels");
    size_t kernel_height = this->desc_.template get_param<size_t>("kernel_height");
    size_t kernel_width  = this->desc_.template get_param<size_t>("kernel_width");
    size_t stride        = this->desc_.template get_param<size_t>("stride");
    size_t padding       = this->desc_.template get_param<size_t>("padding");

    // Get input dimensions
    size_t height = input.dims()[2];
    size_t width  = input.dims()[3];

    // Calculate output dimensions
    auto [output_height, output_width] = get_output_dims(input);

    // Initialize output tensor
    TensorWrapper<T> output({batch_size, out_channels, output_height, output_width});

    // Check if the kernel is 3x3 - our tiled implementation is optimized for this case
    if (kernel_height != 3 || kernel_width != 3) {
        // Fall back to basic implementation for non-3x3 kernels
        return forward_basic(input);
    }

    // Calculate grid dimensions
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((output_width + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (output_height + BLOCK_SIZE - 1) / BLOCK_SIZE,
              batch_size * out_channels);

    // Launch kernel
    conv2d_tiled_kernel<<<grid, block, 0, stream_>>>(input.tensor_data().data,
                                                     weights_.tensor_data().data,
                                                     output.tensor_data().data,
                                                     batch_size,
                                                     in_channels,
                                                     out_channels,
                                                     height,
                                                     width,
                                                     kernel_height,
                                                     kernel_width,
                                                     padding,
                                                     padding,
                                                     stride,
                                                     stride,
                                                     output_height,
                                                     output_width);

    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error("CUDA kernel launch failed: " +
                                 std::string(cudaGetErrorString(error)));
    }

    return output;
}

// cuBLAS-based implementation
template <typename T>
TensorWrapper<T> Conv2D<T>::forward_cublas(const TensorWrapper<T>& input) const {
    // Extract parameters
    size_t batch_size    = input.dims()[0];
    size_t in_channels   = this->desc_.template get_param<size_t>("in_channels");
    size_t out_channels  = this->desc_.template get_param<size_t>("out_channels");
    size_t kernel_height = this->desc_.template get_param<size_t>("kernel_height");
    size_t kernel_width  = this->desc_.template get_param<size_t>("kernel_width");
    size_t stride        = this->desc_.template get_param<size_t>("stride");
    size_t padding       = this->desc_.template get_param<size_t>("padding");

    // Get output dimensions
    auto [output_height, output_width] = get_output_dims(input);

    // Initialize output tensor
    TensorWrapper<T> output({batch_size, out_channels, output_height, output_width});

    // Create im2col tensor
    size_t           gemm_k = in_channels * kernel_height * kernel_width;
    TensorWrapper<T> col_buffer({batch_size, gemm_k, output_height * output_width});

    // Launch im2col kernel (same as basic implementation)
    const int threads_per_block = 256;
    const int total_elements =
        in_channels * kernel_height * kernel_width * output_height * output_width;
    const int blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    // For each batch
    for (size_t n = 0; n < batch_size; ++n) {
        // Perform im2col
        const T* input_data =
            input.tensor_data().data + n * in_channels * input.dims()[2] * input.dims()[3];
        T* col_data = col_buffer.tensor_data().data + n * gemm_k * output_height * output_width;

        im2col_kernel<<<blocks, threads_per_block, 0, stream_>>>(input_data,
                                                                 in_channels,
                                                                 input.dims()[2],
                                                                 input.dims()[3],
                                                                 kernel_height,
                                                                 kernel_width,
                                                                 padding,
                                                                 padding,
                                                                 stride,
                                                                 stride,
                                                                 output_height,
                                                                 output_width,
                                                                 col_data);

        // Perform GEMM using cuBLAS
        T* output_data =
            output.tensor_data().data + n * out_channels * output_height * output_width;

        // Use BLAS with cuBLAS implementation
        BLAS<T>::gemm(false,
                      false,
                      out_channels,
                      output_height * output_width,
                      gemm_k,
                      T(1.0),
                      weights_.tensor_data().data,
                      gemm_k,
                      col_data,
                      output_height * output_width,
                      T(0.0),
                      output_data,
                      output_height * output_width,
                      stream_);
    }

    return output;
}

// Winograd algorithm for F(2x2, 3x3) - optimized for 3x3 kernels
template <typename T>
__global__ void winograd_transform_kernel(const T* input,
                                          int      batch_size,
                                          int      channels,
                                          int      height,
                                          int      width,
                                          int      output_h,
                                          int      output_w,
                                          T*       transformed_input) {
    // Transformation matrices for F(2x2, 3x3)
    // B^T = [1, 0, -1, 0; 0, 1, 1, 0; 0, -1, 1, 0; 0, 1, 0, -1]

    int idx         = blockIdx.x * blockDim.x + threadIdx.x;
    int total_tiles = batch_size * channels * ((output_h + 1) / 2) * ((output_w + 1) / 2);

    if (idx < total_tiles) {
        // Calculate position
        int tile_idx = idx % ((output_h + 1) / 2 * (output_w + 1) / 2);
        int channel  = (idx / ((output_h + 1) / 2 * (output_w + 1) / 2)) % channels;
        int batch    = idx / (channels * ((output_h + 1) / 2 * (output_w + 1) / 2));

        int tile_h = tile_idx / ((output_w + 1) / 2);
        int tile_w = tile_idx % ((output_w + 1) / 2);

        // Input position (top-left of 4x4 tile)
        int h_in = tile_h * 2 - 1; // -1 for padding
        int w_in = tile_w * 2 - 1; // -1 for padding

        // Load 4x4 input tile (with padding)
        T d[4][4] = {0};
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                int h = h_in + i;
                int w = w_in + j;
                if (h >= 0 && h < height && w >= 0 && w < width) {
                    d[i][j] = input[((batch * channels + channel) * height + h) * width + w];
                }
            }
        }

        // Apply Winograd input transformation: B^T * d * B
        T d_tilde[4][4];

        // For each row, compute B^T * d
        for (int i = 0; i < 4; i++) {
            T t0 = d[i][0] - d[i][2];
            T t1 = d[i][1] + d[i][2];
            T t2 = -d[i][1] + d[i][2];
            T t3 = d[i][1] - d[i][3];

            d_tilde[0][i] = t0;
            d_tilde[1][i] = t1;
            d_tilde[2][i] = t2;
            d_tilde[3][i] = t3;
        }

        // For each transformed row, compute result * B
        T result[4][4];
        for (int i = 0; i < 4; i++) {
            T t0 = d_tilde[i][0] - d_tilde[i][2];
            T t1 = d_tilde[i][1] + d_tilde[i][2];
            T t2 = -d_tilde[i][1] + d_tilde[i][2];
            T t3 = d_tilde[i][1] - d_tilde[i][3];

            result[i][0] = t0;
            result[i][1] = t1;
            result[i][2] = t2;
            result[i][3] = t3;
        }

        // Store the transformed input
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                int out_idx =
                    (((batch * channels + channel) * 4 * 4 + i * 4 + j) * ((output_h + 1) / 2) +
                     tile_h) *
                        ((output_w + 1) / 2) +
                    tile_w;
                transformed_input[out_idx] = result[i][j];
            }
        }
    }
}

// Winograd weight transformation
template <typename T>
__global__ void winograd_transform_weights(const T* kernel,
                                           int      out_channels,
                                           int      in_channels,
                                           T*       transformed_weights) {
    // Transformation matrix G = [1, 0, 0; 0.5, 0.5, 0.5; 0.5, -0.5, 0.5; 0, 0, 1]

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < out_channels * in_channels) {
        int in_c  = idx % in_channels;
        int out_c = idx / in_channels;

        // Load 3x3 kernel
        T g[3][3];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                g[i][j] = kernel[((out_c * in_channels + in_c) * 3 + i) * 3 + j];
            }
        }

        // Apply Winograd kernel transformation: G * g * G^T
        T g_tilde[4][4];

        // For each row, compute G * g
        for (int i = 0; i < 3; i++) {
            T gg0 = g[i][0];
            T gg1 = g[i][1];
            T gg2 = g[i][2];

            g_tilde[0][i] = gg0;
            g_tilde[1][i] = (gg0 + gg1 + gg2) * 0.5f;
            g_tilde[2][i] = (gg0 - gg1 + gg2) * 0.5f;
            g_tilde[3][i] = gg2;
        }

        // Special case for i=3 (last row of G * g is all zeros except last column)
        g_tilde[0][3] = 0;
        g_tilde[1][3] = 0;
        g_tilde[2][3] = 0;
        g_tilde[3][3] = 0;

        // For each transformed row, compute result * G^T
        T result[4][4];
        for (int i = 0; i < 4; i++) {
            T gg0 = g_tilde[i][0];
            T gg1 = g_tilde[i][1];
            T gg2 = g_tilde[i][2];

            result[i][0] = gg0;
            result[i][1] = (gg0 + gg1 + gg2) * 0.5f;
            result[i][2] = (gg0 - gg1 + gg2) * 0.5f;
            result[i][3] = gg2;
        }

        // Store the transformed kernel
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                transformed_weights[(((out_c * in_channels + in_c) * 4) + i) * 4 + j] =
                    result[i][j];
            }
        }
    }
}

// Winograd output transformation
template <typename T>
__global__ void winograd_transform_output(const T* transformed_output,
                                          int      batch_size,
                                          int      channels,
                                          int      output_h,
                                          int      output_w,
                                          T*       output) {
    // Transformation matrix A^T = [1, 1, 1, 0; 0, 1, -1, -1]

    int idx         = blockIdx.x * blockDim.x + threadIdx.x;
    int total_tiles = batch_size * channels * ((output_h + 1) / 2) * ((output_w + 1) / 2);

    if (idx < total_tiles) {
        // Calculate position
        int tile_idx = idx % ((output_h + 1) / 2 * (output_w + 1) / 2);
        int channel  = (idx / ((output_h + 1) / 2 * (output_w + 1) / 2)) % channels;
        int batch    = idx / (channels * ((output_h + 1) / 2 * (output_w + 1) / 2));

        int tile_h = tile_idx / ((output_w + 1) / 2);
        int tile_w = tile_idx % ((output_w + 1) / 2);

        // Load 4x4 transformed output
        T m[4][4];
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                int idx =
                    (((batch * channels + channel) * 4 * 4 + i * 4 + j) * ((output_h + 1) / 2) +
                     tile_h) *
                        ((output_w + 1) / 2) +
                    tile_w;
                m[i][j] = transformed_output[idx];
            }
        }

        // Apply Winograd output transformation: A^T * m * A
        T y[2][2];

        // For each transformed row, compute A^T * m
        T temp[2][4];
        for (int i = 0; i < 4; i++) {
            temp[0][i] = m[0][i] + m[1][i] + m[2][i];
            temp[1][i] = m[1][i] - m[2][i] - m[3][i];
        }

        // For each transformed result, compute result * A
        for (int i = 0; i < 2; i++) {
            y[i][0] = temp[i][0] + temp[i][1] + temp[i][2];
            y[i][1] = temp[i][1] - temp[i][2] - temp[i][3];
        }

        // Write to output (if within bounds)
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                int h_out = tile_h * 2 + i;
                int w_out = tile_w * 2 + j;
                if (h_out < output_h && w_out < output_w) {
                    output[((batch * channels + channel) * output_h + h_out) * output_w + w_out] =
                        y[i][j];
                }
            }
        }
    }
}

template <typename T>
TensorWrapper<T> Conv2D<T>::forward_winograd(const TensorWrapper<T>& input) const {
    // This implementation is specialized for 3x3 kernels
    size_t kernel_height = this->desc_.template get_param<size_t>("kernel_height");
    size_t kernel_width  = this->desc_.template get_param<size_t>("kernel_width");

    // Verify this is a 3x3 kernel
    if (kernel_height != 3 || kernel_width != 3) {
        // Fallback to cuBLAS for non-3x3 kernels
        return forward_cublas(input);
    }

    // Extract parameters
    size_t batch_size   = input.dims()[0];
    size_t in_channels  = this->desc_.template get_param<size_t>("in_channels");
    size_t out_channels = this->desc_.template get_param<size_t>("out_channels");
    size_t stride       = this->desc_.template get_param<size_t>("stride");
    size_t padding      = this->desc_.template get_param<size_t>("padding");

    // If stride != 1, fallback to cuBLAS (Winograd requires stride=1)
    if (stride != 1) {
        return forward_cublas(input);
    }

    // Get input and output dimensions
    size_t height                      = input.dims()[2];
    size_t width                       = input.dims()[3];
    auto [output_height, output_width] = get_output_dims(input);

    // Calculate number of tiles
    size_t tiles_h   = (output_height + 1) / 2;
    size_t tiles_w   = (output_width + 1) / 2;
    size_t num_tiles = tiles_h * tiles_w;

    // Create output tensor
    TensorWrapper<T> output({batch_size, out_channels, output_height, output_width});

    // Allocate transformation buffers
    TensorWrapper<T> transformed_input({batch_size, in_channels, 4 * 4, num_tiles});
    TensorWrapper<T> transformed_weights({out_channels, in_channels, 4, 4});
    TensorWrapper<T> transformed_output({batch_size, out_channels, 4 * 4, num_tiles});

    // Transform weights (can be precomputed and stored)
    int threads_per_block = 256;
    int weight_blocks = (out_channels * in_channels + threads_per_block - 1) / threads_per_block;

    winograd_transform_weights<<<weight_blocks, threads_per_block, 0, stream_>>>(
        weights_.tensor_data().data,
        out_channels,
        in_channels,
        transformed_weights.tensor_data().data);

    // Transform input
    int total_input_tiles = batch_size * in_channels * num_tiles;
    int input_blocks      = (total_input_tiles + threads_per_block - 1) / threads_per_block;

    winograd_transform_kernel<<<input_blocks, threads_per_block, 0, stream_>>>(
        input.tensor_data().data,
        batch_size,
        in_channels,
        height,
        width,
        output_height,
        output_width,
        transformed_input.tensor_data().data);

    // Element-wise multiplication (batch matmul)
    // This can be done using GEMM by reshaping the data
    for (size_t n = 0; n < batch_size; n++) {
        for (size_t tile = 0; tile < 16; tile++) { // 4x4 = 16 elements per tile
            // Extract the appropriate slice for this tile element
            const T* input_slice = transformed_input.tensor_data().data +
                                   n * in_channels * 16 * num_tiles + tile * num_tiles;

            const T* weight_slice = transformed_weights.tensor_data().data + tile;

            T* output_slice = transformed_output.tensor_data().data +
                              n * out_channels * 16 * num_tiles + tile * num_tiles;

            // Perform matrix multiplication C = A * B
            // A: [out_channels, in_channels]
            // B: [in_channels, num_tiles]
            // C: [out_channels, num_tiles]
            BLAS<T>::gemm(false,
                          false,
                          out_channels,
                          num_tiles,
                          in_channels,
                          T(1.0),
                          weight_slice,
                          in_channels * 16, // stride of weight matrix
                          input_slice,
                          num_tiles, // stride of input matrix
                          T(0.0),
                          output_slice,
                          num_tiles, // stride of output matrix
                          stream_);
        }
    }

    // Transform output
    int total_output_tiles = batch_size * out_channels * num_tiles;
    int output_blocks      = (total_output_tiles + threads_per_block - 1) / threads_per_block;

    winograd_transform_output<<<output_blocks, threads_per_block, 0, stream_>>>(
        transformed_output.tensor_data().data,
        batch_size,
        out_channels,
        output_height,
        output_width,
        output.tensor_data().data);

    return output;
}

// Factory function implementation
template <typename T>
std::unique_ptr<BaseKernel<T>> make_conv2d(size_t kernel_height,
                                           size_t kernel_width,
                                           size_t in_channels,
                                           size_t out_channels,
                                           size_t stride,
                                           size_t padding) {
    // Use the correct enum value from your KernelType
    KernelDescriptor desc(KernelType::Convolution2D);

    // Set parameters using set_param
    desc.set_param("kernel_height", kernel_height);
    desc.set_param("kernel_width", kernel_width);
    desc.set_param("in_channels", in_channels);
    desc.set_param("out_channels", out_channels);
    desc.set_param("stride", stride);
    desc.set_param("padding", padding);

    // Create and return the Conv2D kernel
    return std::make_unique<Conv2D<T>>(desc);
}

// Explicit template instantiations to avoid linker errors
template class Conv2D<float>;
template class Conv2D<double>;

// Explicit instantiation of factory functions
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

} // namespace hpc::compute::cuda