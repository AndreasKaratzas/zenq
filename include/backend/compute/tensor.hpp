#pragma once

#include "backend/common/types.hpp"
#include <memory>

#ifndef __CUDACC__
    #include <span>
#endif

namespace hpc {
namespace compute {

class Tensor {
public:
    // Constructors and destructor
    Tensor(const hpc::core::Shape& shape, hpc::core::DataType dtype);
    Tensor(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    ~Tensor();

    // Assignment operators
    Tensor& operator=(const Tensor& other);
    Tensor& operator=(Tensor&& other) noexcept;

    // Accessors
    const hpc::core::Shape& shape() const noexcept {
        return shape_;
    }
    hpc::core::DataType dtype() const noexcept {
        return dtype_;
    }
    size_t size() const noexcept {
        return total_size_;
    }
    void* data() noexcept {
        return data_;
    }
    const void* data() const noexcept {
        return data_;
    }

    // Memory management
    void resize(const hpc::core::Shape& new_shape);
    void zero_();

#ifndef __CUDACC__
    // Type-safe data access with span (CPU only)
    template <typename T>
    std::span<T> data_span() {
        if (get_data_type<T>() != dtype_) {
            throw std::runtime_error("Type mismatch in data_span");
        }
        return std::span<T>(static_cast<T*>(data_), total_size_);
    }

    template <typename T>
    std::span<const T> data_span() const {
        if (get_data_type<T>() != dtype_) {
            throw std::runtime_error("Type mismatch in data_span");
        }
        return std::span<const T>(static_cast<const T*>(data_), total_size_);
    }
#endif

    // Type-safe data access with pointers (both CPU and CUDA)
    template <typename T>
    T* data() {
        if (get_data_type<T>() != dtype_) {
            throw std::runtime_error("Type mismatch in data access");
        }
        return static_cast<T*>(data_);
    }

    template <typename T>
    const T* data() const {
        if (get_data_type<T>() != dtype_) {
            throw std::runtime_error("Type mismatch in data access");
        }
        return static_cast<const T*>(data_);
    }

    // Size-aware data access (alternative to span for CUDA)
    template <typename T>
    struct DataView {
        T*     ptr;
        size_t size;
    };

    template <typename T>
    DataView<T> get_data_view() {
        if (get_data_type<T>() != dtype_) {
            throw std::runtime_error("Type mismatch in data view");
        }
        return DataView<T>{static_cast<T*>(data_), total_size_};
    }

    template <typename T>
    DataView<const T> get_data_view() const {
        if (get_data_type<T>() != dtype_) {
            throw std::runtime_error("Type mismatch in data view");
        }
        return DataView<const T>{static_cast<const T*>(data_), total_size_};
    }

private:
    void   allocate_memory();
    void   deallocate_memory();
    size_t calculate_size() const;

    // Helper function to get DataType from C++ type
    template <typename T>
    static hpc::core::DataType get_data_type() {
        if constexpr (std::is_same_v<T, float>) {
            return hpc::core::DataType::FLOAT32;
        } else if constexpr (std::is_same_v<T, double>) {
            return hpc::core::DataType::FLOAT64;
        } else if constexpr (std::is_same_v<T, int32_t>) {
            return hpc::core::DataType::INT32;
        } else if constexpr (std::is_same_v<T, int64_t>) {
            return hpc::core::DataType::INT64;
        } else {
            throw std::runtime_error("Unsupported data type");
        }
    }

    hpc::core::Shape    shape_;
    size_t              total_size_;
    void*               data_;
    hpc::core::DataType dtype_;
    bool                owns_data_;

    static constexpr size_t ALIGNMENT = 64;
};

} // namespace compute
} // namespace hpc