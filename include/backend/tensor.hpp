#pragma once

#include "common/types.hpp"
#include <memory>
#include <span>

namespace kernel_compute {

class Tensor {
public:
    Tensor() = default;

    // Create tensor with shape and type
    Tensor(const Shape& shape, DataType dtype);

    // Create tensor with data
    template <typename T>
    Tensor(const Shape& shape, DataType dtype, std::span<const T> data);

    // Move operations
    Tensor(Tensor&&) noexcept            = default;
    Tensor& operator=(Tensor&&) noexcept = default;

    // Copy operations (disabled by default)
    Tensor(const Tensor&)            = delete;
    Tensor& operator=(const Tensor&) = delete;

    // Enable copy when explicitly requested
    Tensor clone() const;

    // Accessors
    const Shape& shape() const noexcept {
        return shape_;
    }
    DataType dtype() const noexcept {
        return dtype_;
    }
    std::size_t size() const noexcept {
        return size_;
    }
    bool empty() const noexcept {
        return size_ == 0;
    }

    // Data access
    template <typename T>
    std::span<T> data() {
        return std::span<T>(static_cast<T*>(data_.get()), size_);
    }

    template <typename T>
    std::span<const T> data() const {
        return std::span<const T>(static_cast<const T*>(data_.get()), size_);
    }

    // Memory management
    void to_device(DeviceType device);
    void to_host();

private:
    Shape                 shape_;
    DataType              dtype_{DataType::Float32};
    std::size_t           size_{0};
    std::shared_ptr<void> data_;
    DeviceType            current_device_{DeviceType::CPU};

    std::size_t compute_size() const;
    std::size_t element_size() const;
};

} // namespace kernel_compute