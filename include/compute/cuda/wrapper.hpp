#pragma once

#include "compute/cpp/tensor.hpp"
#include "compute/cuda/tensor.cuh"
#include <vector>

namespace hpc::compute::cuda {

template <typename T>
class TensorWrapper {
public:
    using HostTensor = hpc::compute::Tensor<T>;

    // Constructor for empty tensor
    TensorWrapper() : tensor_data_(impl::create_tensor<T>(nullptr, 0)) {}

    // Constructor with dimensions
    explicit TensorWrapper(const std::vector<std::size_t>& dims,
                           MemoryLayout                    layout = MemoryLayout::RowMajor)
        : tensor_data_(impl::create_tensor<T>(dims.data(), dims.size(), layout)) {}

    // Constructor from host tensor
    explicit TensorWrapper(const HostTensor& host_tensor)
        : TensorWrapper(get_dims(host_tensor), convert_layout(host_tensor.layout())) {
        copy_from_host(host_tensor);
    }

    ~TensorWrapper() {
        impl::destroy_tensor(tensor_data_);
    }

    // Move operations
    TensorWrapper(TensorWrapper&& other) noexcept : tensor_data_(other.tensor_data_) {
        other.tensor_data_.data = nullptr;
        other.tensor_data_.size = 0;
        other.tensor_data_.rank = 0;
    }

    TensorWrapper& operator=(TensorWrapper&& other) noexcept {
        if (this != &other) {
            impl::destroy_tensor(tensor_data_);
            tensor_data_            = other.tensor_data_;
            other.tensor_data_.data = nullptr;
            other.tensor_data_.size = 0;
            other.tensor_data_.rank = 0;
        }
        return *this;
    }

    // Delete copy operations
    TensorWrapper(const TensorWrapper&)            = delete;
    TensorWrapper& operator=(const TensorWrapper&) = delete;

    // Host data transfer
    void copy_from_host(const HostTensor& host_tensor) {
        impl::copy_to_device(tensor_data_, host_tensor.data(), host_tensor.size());
    }

    void copy_to_host(HostTensor& host_tensor) const {
        if (host_tensor.size() != tensor_data_.size) {
            host_tensor = HostTensor(get_dims(), convert_layout_back(tensor_data_.layout));
        }
        impl::copy_to_host(tensor_data_, host_tensor.data(), host_tensor.size());
    }

    // Access to raw CUDA data
    const TensorData<T>& tensor_data() const noexcept {
        return tensor_data_;
    }
    TensorData<T>& tensor_data() noexcept {
        return tensor_data_;
    }

    // Utility functions
    void zero() {
        impl::zero_tensor(tensor_data_);
    }
    bool is_contiguous() const {
        return impl::is_contiguous(tensor_data_);
    }
    std::size_t size() const noexcept {
        return tensor_data_.size;
    }
    std::size_t rank() const noexcept {
        return tensor_data_.rank;
    }
    const std::array<std::size_t, TensorData<T>::MAX_DIMS>& dims() const noexcept {
        return tensor_data_.dims;
    }

private:
    TensorData<T> tensor_data_;

    static std::vector<std::size_t> get_dims(const HostTensor& host_tensor) {
        return std::vector<std::size_t>(host_tensor.dims().begin(),
                                        host_tensor.dims().begin() + host_tensor.rank());
    }

    std::vector<std::size_t> get_dims() const {
        return std::vector<std::size_t>(tensor_data_.dims.begin(),
                                        tensor_data_.dims.begin() + tensor_data_.rank);
    }

    static MemoryLayout convert_layout(typename HostTensor::Layout layout) {
        return layout == HostTensor::Layout::RowMajor ? MemoryLayout::RowMajor
                                                      : MemoryLayout::ColumnMajor;
    }

    static typename HostTensor::Layout convert_layout_back(MemoryLayout layout) {
        return layout == MemoryLayout::RowMajor ? HostTensor::Layout::RowMajor
                                                : HostTensor::Layout::ColumnMajor;
    }
};

} // namespace hpc::compute::cuda