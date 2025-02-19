#include "backend/common/tensor.hpp"
#include <cuda_runtime.h>
#include <stdexcept>

namespace kernel_compute {

Tensor::Tensor(const Shape& shape, DataType dtype)
    : shape_(shape), dtype_(dtype), size_(compute_size()) {
    const auto total_size = size_ * element_size();
    data_                 = std::shared_ptr<void>(malloc(total_size), free);
    if (!data_) {
        throw std::runtime_error("Failed to allocate tensor memory");
    }
}

template <typename T>
Tensor::Tensor(const Shape& shape, DataType dtype, std::span<const T> data) : Tensor(shape, dtype) {
    if (data.size() != size_) {
        throw std::invalid_argument("Data size doesn't match tensor shape");
    }
    std::memcpy(data_.get(), data.data(), size_ * element_size());
}

Tensor Tensor::clone() const {
    Tensor copy(shape_, dtype_);
    std::memcpy(copy.data_.get(), data_.get(), size_ * element_size());
    copy.current_device_ = current_device_;
    return copy;
}

void Tensor::to_device(DeviceType device) {
    if (device == current_device_)
        return;

    if (device == DeviceType::CUDA) {
        void* device_ptr;
        if (cudaMalloc(&device_ptr, size_ * element_size()) != cudaSuccess) {
            throw std::runtime_error("CUDA memory allocation failed");
        }
        if (cudaMemcpy(device_ptr, data_.get(), size_ * element_size(), cudaMemcpyHostToDevice) !=
            cudaSuccess) {
            cudaFree(device_ptr);
            throw std::runtime_error("CUDA memory copy failed");
        }
        data_           = std::shared_ptr<void>(device_ptr, cudaFree);
        current_device_ = DeviceType::CUDA;
    } else {
        to_host();
    }
}

void Tensor::to_host() {
    if (current_device_ == DeviceType::CPU)
        return;

    void* host_ptr = malloc(size_ * element_size());
    if (!host_ptr) {
        throw std::runtime_error("Host memory allocation failed");
    }
    if (cudaMemcpy(host_ptr, data_.get(), size_ * element_size(), cudaMemcpyDeviceToHost) !=
        cudaSuccess) {
        free(host_ptr);
        throw std::runtime_error("CUDA to host memory copy failed");
    }
    data_           = std::shared_ptr<void>(host_ptr, free);
    current_device_ = DeviceType::CPU;
}

std::size_t Tensor::compute_size() const {
    if (shape_.empty())
        return 0;
    size_t total = 1;
    for (const auto dim : shape_) {
        total *= dim;
    }
    return total;
}

std::size_t Tensor::element_size() const {
    switch (dtype_) {
    case DataType::Float32:
        return sizeof(float32_t);
    case DataType::Float64:
        return sizeof(float64_t);
    case DataType::Int32:
        return sizeof(int32_t);
    case DataType::Int64:
        return sizeof(int64_t);
    default:
        throw std::runtime_error("Unknown data type");
    }
}

// Explicit template instantiations
template Tensor::Tensor(const Shape&, DataType, std::span<const float32_t>);
template Tensor::Tensor(const Shape&, DataType, std::span<const float64_t>);
template Tensor::Tensor(const Shape&, DataType, std::span<const int32_t>);
template Tensor::Tensor(const Shape&, DataType, std::span<const int64_t>);

} // namespace kernel_compute