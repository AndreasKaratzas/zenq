// tensor.cpp
#include "backend/compute/tensor.hpp"
#include <cstring>
#include <numeric>
#include <stdexcept>

namespace hpc {
namespace compute {

Tensor::Tensor(const hpc::core::Shape& shape, hpc::core::DataType dtype)
    : shape_(shape), dtype_(dtype), owns_data_(true) {
    total_size_ = calculate_size();
    allocate_memory();
}

Tensor::Tensor(const Tensor& other)
    : shape_(other.shape_), total_size_(other.total_size_), dtype_(other.dtype_), owns_data_(true) {
    allocate_memory();
    std::memcpy(data_, other.data_, total_size_ * sizeof(float));
}

Tensor::Tensor(Tensor&& other) noexcept
    : shape_(std::move(other.shape_)), total_size_(other.total_size_), data_(other.data_),
      dtype_(other.dtype_), owns_data_(other.owns_data_) {
    other.data_      = nullptr;
    other.owns_data_ = false;
}

Tensor::~Tensor() {
    if (owns_data_) {
        deallocate_memory();
    }
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        if (owns_data_) {
            deallocate_memory();
        }
        shape_      = other.shape_;
        total_size_ = other.total_size_;
        dtype_      = other.dtype_;
        owns_data_  = true;
        allocate_memory();
        std::memcpy(data_, other.data_, total_size_ * sizeof(float));
    }
    return *this;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        if (owns_data_) {
            deallocate_memory();
        }
        shape_           = std::move(other.shape_);
        total_size_      = other.total_size_;
        data_            = other.data_;
        dtype_           = other.dtype_;
        owns_data_       = other.owns_data_;
        other.data_      = nullptr;
        other.owns_data_ = false;
    }
    return *this;
}

void Tensor::resize(const hpc::core::Shape& new_shape) {
    shape_          = new_shape;
    size_t new_size = calculate_size();
    if (new_size != total_size_) {
        if (owns_data_) {
            deallocate_memory();
        }
        total_size_ = new_size;
        allocate_memory();
    }
}

void Tensor::zero_() {
    std::memset(data_, 0, total_size_ * sizeof(float));
}

void Tensor::allocate_memory() {
    size_t element_size;
    switch (dtype_) {
    case hpc::core::DataType::FLOAT32:
        element_size = sizeof(float);
        break;
    case hpc::core::DataType::FLOAT64:
        element_size = sizeof(double);
        break;
    case hpc::core::DataType::INT32:
        element_size = sizeof(int32_t);
        break;
    case hpc::core::DataType::INT64:
        element_size = sizeof(int64_t);
        break;
    default:
        throw std::runtime_error("Unsupported data type");
    }

    size_t aligned_size = ((total_size_ * element_size + ALIGNMENT - 1) / ALIGNMENT) * ALIGNMENT;
    data_               = aligned_alloc(ALIGNMENT, aligned_size);
    if (!data_) {
        throw std::runtime_error("Failed to allocate memory");
    }
}

void Tensor::deallocate_memory() {
    if (data_) {
        free(data_);
        data_ = nullptr;
    }
}

size_t Tensor::calculate_size() const {
    return std::accumulate(shape_.begin(), shape_.end(), 1ULL, std::multiplies<size_t>());
}

} // namespace compute
} // namespace hpc