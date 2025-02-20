#pragma once

#include <array>
#include <cstddef>
#include <cuda_runtime.h>
#include <type_traits>
#include <vector>

namespace hpc::compute::cuda {

enum class MemoryLayout {
    RowMajor,   // C-style, last dimension varies fastest
    ColumnMajor // Fortran-style, first dimension varies fastest
};

template <typename T>
struct TensorData {
    static_assert(std::is_arithmetic_v<T>, "Tensor type must be numeric");
    static constexpr std::size_t MAX_DIMS = 8;

    T*                                data;
    std::array<std::size_t, MAX_DIMS> dims;
    std::array<std::size_t, MAX_DIMS> strides;
    std::size_t                       rank;
    std::size_t                       size;
    MemoryLayout                      layout;
};

namespace impl {

template <typename T>
__host__ TensorData<T> create_tensor(const std::size_t* dims,
                                     std::size_t        rank,
                                     MemoryLayout       layout = MemoryLayout::RowMajor);

template <typename T>
__host__ void destroy_tensor(TensorData<T>& tensor);

template <typename T>
__host__ void copy_to_device(TensorData<T>& tensor, const T* host_data, std::size_t size);

template <typename T>
__host__ void copy_to_host(const TensorData<T>& tensor, T* host_data, std::size_t size);

template <typename T>
__host__ void zero_tensor(TensorData<T>& tensor);

template <typename T>
__host__ __device__ std::size_t compute_offset(const TensorData<T>& tensor,
                                               const std::size_t*   indices);

template <typename T>
__host__ __device__ bool is_contiguous(const TensorData<T>& tensor);

template <typename T>
__host__ __device__ void compute_strides(TensorData<T>& tensor);

void check_cuda_error(cudaError_t error, const char* file, int line);
} // namespace impl

} // namespace hpc::compute::cuda