#pragma once

#include "compute/cpp/tensor.hpp"
#include "compute/cuda/tensor.cuh"
#include <initializer_list>
#include <vector>

namespace hpc::compute::cuda {

template <typename T>
class TensorWrapper {
public:
    using HostTensor = hpc::compute::Tensor<T>;

    // Constructor for empty tensor
    TensorWrapper() : tensor_data_(impl::create_tensor<T>(nullptr, 0)) {}

    // Constructor with initializer list
    TensorWrapper(std::initializer_list<std::size_t> dims,
                  MemoryLayout                       layout = MemoryLayout::RowMajor)
        : TensorWrapper(std::vector<std::size_t>(dims.begin(), dims.end()), layout) {}

    // Constructor with dimensions vector
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
        other.tensor_data_.dims.fill(0);
        other.tensor_data_.strides.fill(0);
    }

    TensorWrapper& operator=(TensorWrapper&& other) noexcept {
        if (this != &other) {
            impl::destroy_tensor(tensor_data_);
            tensor_data_            = other.tensor_data_;
            other.tensor_data_.data = nullptr;
            other.tensor_data_.size = 0;
            other.tensor_data_.rank = 0;
            other.tensor_data_.dims.fill(0);
            other.tensor_data_.strides.fill(0);
        }
        return *this;
    }

    T operator()(size_t i, size_t j) const {
        if (tensor_data_.layout == MemoryLayout::RowMajor) {
            return tensor_data_.data[i * tensor_data_.dims[1] + j];
        } else {
            return tensor_data_.data[j * tensor_data_.dims[0] + i];
        }
    }

    // Delete copy operations
    TensorWrapper(const TensorWrapper&)            = delete;
    TensorWrapper& operator=(const TensorWrapper&) = delete;

    // Host data transfer
    void copy_from_host(const HostTensor& host_tensor) {
        // If layouts match, do direct copy
        if ((tensor_data_.layout == MemoryLayout::RowMajor &&
             host_tensor.layout() == hpc::compute::MemoryLayout::RowMajor) ||
            (tensor_data_.layout == MemoryLayout::ColumnMajor &&
             host_tensor.layout() == hpc::compute::MemoryLayout::ColumnMajor)) {
            impl::copy_to_device(tensor_data_, host_tensor.data(), host_tensor.size());
            return;
        }

        // For layout conversion
        std::vector<T> temp(host_tensor.size());
        const size_t   rows = tensor_data_.dims[0];
        const size_t   cols = tensor_data_.dims[1];

        // Convert from row-major to column-major
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                temp[j * rows + i] = host_tensor.data()[i * cols + j];
            }
        }

        impl::copy_to_device(tensor_data_, temp.data(), temp.size());
    }

    void copy_to_host(HostTensor& host_tensor) const {
        if (host_tensor.size() != tensor_data_.size) {
            host_tensor = HostTensor(get_dims(), convert_layout_back(tensor_data_.layout));
        }

        // If layouts match, do direct copy
        if ((tensor_data_.layout == MemoryLayout::RowMajor &&
             host_tensor.layout() == hpc::compute::MemoryLayout::RowMajor) ||
            (tensor_data_.layout == MemoryLayout::ColumnMajor &&
             host_tensor.layout() == hpc::compute::MemoryLayout::ColumnMajor)) {
            impl::copy_to_host(tensor_data_, host_tensor.data(), host_tensor.size());
            return;
        }

        // For layout conversion
        std::vector<T> temp(tensor_data_.size);
        impl::copy_to_host(tensor_data_, temp.data(), temp.size());

        const size_t rows = tensor_data_.dims[0];
        const size_t cols = tensor_data_.dims[1];

        // Convert from column-major to row-major
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                host_tensor.data()[i * cols + j] = temp[j * rows + i];
            }
        }
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
    [[nodiscard]] bool is_contiguous() const {
        return impl::is_contiguous(tensor_data_);
    }
    [[nodiscard]] std::size_t size() const noexcept {
        return tensor_data_.size;
    }
    [[nodiscard]] std::size_t rank() const noexcept {
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

    [[nodiscard]] std::vector<std::size_t> get_dims() const {
        return std::vector<std::size_t>(tensor_data_.dims.begin(),
                                        tensor_data_.dims.begin() + tensor_data_.rank);
    }

    static MemoryLayout convert_layout(hpc::compute::MemoryLayout layout) {
        return layout == hpc::compute::MemoryLayout::RowMajor ? MemoryLayout::RowMajor
                                                              : MemoryLayout::ColumnMajor;
    }

    static hpc::compute::MemoryLayout convert_layout_back(MemoryLayout layout) {
        return layout == MemoryLayout::RowMajor ? hpc::compute::MemoryLayout::RowMajor
                                                : hpc::compute::MemoryLayout::ColumnMajor;
    }
};

} // namespace hpc::compute::cuda