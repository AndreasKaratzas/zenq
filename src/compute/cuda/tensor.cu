#include "compute/cuda/tensor.cuh"
#include <stdexcept>

namespace hpc::compute::cuda {
namespace impl {

void check_cuda_error(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error at ") + file + ":" + std::to_string(line) +
                                 ": " + cudaGetErrorString(error));
    }
}

template <typename T>
__host__ TensorData<T> create_tensor(const std::size_t* dims,
                                     std::size_t        rank,
                                     MemoryLayout       layout) {
    TensorData<T> tensor;
    tensor.data   = nullptr;
    tensor.rank   = rank;
    tensor.size   = 1;
    tensor.layout = layout;
    tensor.dims.fill(1);
    tensor.strides.fill(0);

    if (rank > 0) {
        if (rank > TensorData<T>::MAX_DIMS) {
            throw std::invalid_argument("Too many dimensions");
        }

        for (std::size_t i = 0; i < rank; ++i) {
            tensor.dims[i] = dims[i];
            tensor.size *= dims[i];
        }

        compute_strides(tensor);

        if (tensor.size > 0) {
            check_cuda_error(cudaMalloc(&tensor.data, tensor.size * sizeof(T)), __FILE__, __LINE__);
        }
    }

    return tensor;
}

template <typename T>
__host__ void destroy_tensor(TensorData<T>& tensor) {
    if (tensor.data) {
        cudaFree(tensor.data);
        tensor.data = nullptr;
    }
    tensor.rank = 0;
    tensor.size = 0;
    tensor.dims.fill(0);
    tensor.strides.fill(0);
}

template <typename T>
__host__ void copy_to_device(TensorData<T>& tensor, const T* host_data, std::size_t size) {
    if (size != tensor.size) {
        throw std::invalid_argument("Size mismatch in copy_to_device");
    }
    check_cuda_error(cudaMemcpy(tensor.data, host_data, size * sizeof(T), cudaMemcpyHostToDevice),
                     __FILE__,
                     __LINE__);
}

template <typename T>
__host__ void copy_to_host(const TensorData<T>& tensor, T* host_data, std::size_t size) {
    if (size != tensor.size) {
        throw std::invalid_argument("Size mismatch in copy_to_host");
    }
    check_cuda_error(cudaMemcpy(host_data, tensor.data, size * sizeof(T), cudaMemcpyDeviceToHost),
                     __FILE__,
                     __LINE__);
}

template <typename T>
__host__ void zero_tensor(TensorData<T>& tensor) {
    if (tensor.data && tensor.size > 0) {
        check_cuda_error(cudaMemset(tensor.data, 0, tensor.size * sizeof(T)), __FILE__, __LINE__);
    }
}

template <typename T>
__host__ __device__ void compute_strides(TensorData<T>& tensor) {
    if (tensor.layout == MemoryLayout::RowMajor) {
        std::size_t stride = 1;
        for (int i = tensor.rank - 1; i >= 0; --i) {
            tensor.strides[i] = stride;
            stride *= tensor.dims[i];
        }
    } else {
        std::size_t stride = 1;
        for (std::size_t i = 0; i < tensor.rank; ++i) {
            tensor.strides[i] = stride;
            stride *= tensor.dims[i];
        }
    }
}

template <typename T>
__host__ __device__ std::size_t compute_offset(const TensorData<T>& tensor,
                                               const std::size_t*   indices) {
    std::size_t offset = 0;
    for (std::size_t i = 0; i < tensor.rank; ++i) {
#ifndef __CUDA_ARCH__
        if (indices[i] >= tensor.dims[i]) {
            throw std::out_of_range("Index out of bounds");
        }
#endif
        offset += indices[i] * tensor.strides[i];
    }
    return offset;
}

template <typename T>
__host__ __device__ bool is_contiguous(const TensorData<T>& tensor) {
    std::size_t expected_stride = 1;
    for (int i = tensor.rank - 1; i >= 0; --i) {
        if (tensor.strides[i] != expected_stride) {
            return false;
        }
        expected_stride *= tensor.dims[i];
    }
    return true;
}

// Explicit instantiations
#define INSTANTIATE_FOR_TYPE(T)                                                                    \
    template TensorData<T> create_tensor<T>(const std::size_t*, std::size_t, MemoryLayout);        \
    template void          destroy_tensor<T>(TensorData<T>&);                                      \
    template void          copy_to_device<T>(TensorData<T>&, const T*, std::size_t);               \
    template void          copy_to_host<T>(const TensorData<T>&, T*, std::size_t);                 \
    template void          zero_tensor<T>(TensorData<T>&);                                         \
    template void          compute_strides<T>(TensorData<T>&);                                     \
    template std::size_t   compute_offset<T>(const TensorData<T>&, const std::size_t*);            \
    template bool          is_contiguous<T>(const TensorData<T>&);

INSTANTIATE_FOR_TYPE(float)
INSTANTIATE_FOR_TYPE(double)
INSTANTIATE_FOR_TYPE(int)
INSTANTIATE_FOR_TYPE(unsigned int)

#undef INSTANTIATE_FOR_TYPE

} // namespace impl
} // namespace hpc::compute::cuda