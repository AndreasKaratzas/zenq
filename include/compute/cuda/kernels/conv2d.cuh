#pragma once

#include "compute/cpp/kernel.hpp"
#include "compute/cuda/tensor.cuh"
#include "compute/cuda/wrapper.hpp"
#include <cuda_runtime.h>

#define BLOCK_SIZE 16
#define TILE_SIZE 16

namespace hpc::compute::cuda {

// Similar to CPUFeatures, but for CUDA devices
struct CUDAFeatures {
    int    computeCapabilityMajor;
    int    computeCapabilityMinor;
    int    multiProcessorCount;
    int    maxThreadsPerBlock;
    size_t totalGlobalMem;

    static CUDAFeatures detect(int deviceId = 0);
};

// Different implementations for different CUDA capabilities
enum class ConvImplementation {
    Basic,       // Simple CUDA implementation
    Tiled,       // Tiled/shared memory implementation
    CuBLAS,      // Using cuBLAS for GEMM operations
    WinogradFast // Using Winograd fast convolution algorithm
};

// Conv2D class for CUDA
template <typename T>
class Conv2D : public BaseKernel<T> {
public:
    explicit Conv2D(const KernelDescriptor& desc);
    ~Conv2D();

    // Make sure to use 'override' keyword to explicitly mark overridden virtual functions
    void validate_input(const Tensor<T>& input) const override;

    // Override the BaseKernel methods with the correct return types
    [[nodiscard]] Tensor<T> forward(const Tensor<T>& input,
                                    bool             return_to_cpu = true) const override;

    // New method for direct GPU-to-GPU forward pass (no automatic CPU conversion)
    [[nodiscard]] TensorWrapper<T> forward_gpu(const TensorWrapper<T>& input) const;

    void load_weights(Tensor<T>&& weights) override;

    [[nodiscard]] const Tensor<T>& weights() const override {
        // Since there's no direct as_tensor() method, use copy_to_host()
        if (weights_host_.size() == 0) {
            // Lazy initialization if not already done
            const_cast<Conv2D<T>*>(this)->weights_host_ = Tensor<T>(std::vector<std::size_t>(
                weights_.dims().begin(), weights_.dims().begin() + weights_.rank()));
            const_cast<Conv2D<T>*>(this)->weights_.copy_to_host(weights_host_);
        }
        return weights_host_;
    }

    // Utility method to get output dimensions for a given input
    [[nodiscard]] std::vector<size_t> get_output_shape(const Tensor<T>& input) const;

private:
    void                                    initialize_weights();
    [[nodiscard]] std::pair<size_t, size_t> get_output_dims(const TensorWrapper<T>& input) const;

    // Internal implementation can use TensorWrapper for CUDA operations
    TensorWrapper<T>  weights_;      // GPU version
    mutable Tensor<T> weights_host_; // CPU version (mutable to allow lazy init)

    // Internal implementation methods that use TensorWrapper
    [[nodiscard]] TensorWrapper<T> forward_basic(const TensorWrapper<T>& input) const;
    [[nodiscard]] TensorWrapper<T> forward_tiled(const TensorWrapper<T>& input) const;
    [[nodiscard]] TensorWrapper<T> forward_cublas(const TensorWrapper<T>& input) const;
    [[nodiscard]] TensorWrapper<T> forward_winograd(const TensorWrapper<T>& input) const;

    // CUDA-related members
    static const CUDAFeatures        cuda_features_;
    [[nodiscard]] ConvImplementation select_implementation() const;

    // CUDA stream
    cudaStream_t stream_;
};

// Factory function similar to CPU version
template <typename T>
std::unique_ptr<BaseKernel<T>> make_conv2d(size_t kernel_height,
                                           size_t kernel_width,
                                           size_t in_channels,
                                           size_t out_channels,
                                           size_t stride  = 1,
                                           size_t padding = 0);

} // namespace hpc::compute::cuda