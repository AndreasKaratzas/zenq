#pragma once

#include "compute/cpp/tensor.hpp"
#include <array>
#include <cstddef>
#include <utility>

namespace hpc::compute {

template <Numeric T>
class TensorView {
public:
    static constexpr std::size_t MAX_DIMS = 8;
    using value_type                      = T;
    using size_type                       = std::size_t;
    using pointer                         = T*;
    using const_pointer                   = const T*;
    using reference                       = T&;
    using const_reference                 = const T&;

    // Constructor for creating a view from a tensor subset
    TensorView(pointer                                data,
               const std::array<size_type, MAX_DIMS>& dims,
               const std::array<size_type, MAX_DIMS>& strides,
               size_type                              rank,
               MemoryLayout                           layout) noexcept;

    // Data access
    reference       at(const std::array<size_type, MAX_DIMS>& indices);
    const_reference at(const std::array<size_type, MAX_DIMS>& indices) const;

    template <typename... Indices>
    reference operator()(Indices... indices) {
        static_assert(sizeof...(Indices) <= MAX_DIMS, "Too many indices");
        std::array<size_type, MAX_DIMS> idx = {static_cast<size_type>(indices)...};
        return at(idx);
    }

    template <typename... Indices>
    const_reference operator()(Indices... indices) const {
        static_assert(sizeof...(Indices) <= MAX_DIMS, "Too many indices");
        std::array<size_type, MAX_DIMS> idx = {static_cast<size_type>(indices)...};
        return at(idx);
    }

    // Properties
    [[nodiscard]] pointer data() noexcept {
        return data_;
    }
    [[nodiscard]] const_pointer data() const noexcept {
        return data_;
    }
    [[nodiscard]] size_type rank() const noexcept {
        return rank_;
    }
    [[nodiscard]] const std::array<size_type, MAX_DIMS>& dims() const noexcept {
        return dims_;
    }
    [[nodiscard]] const std::array<size_type, MAX_DIMS>& strides() const noexcept {
        return strides_;
    }
    [[nodiscard]] MemoryLayout layout() const noexcept {
        return layout_;
    }
    [[nodiscard]] bool is_contiguous() const noexcept;

    // Create a sub-view
    TensorView<T> view(std::initializer_list<std::pair<size_type, size_type>> ranges);
    TensorView<T> slice(size_type dim, size_type index);

private:
    pointer                         data_;
    std::array<size_type, MAX_DIMS> dims_;
    std::array<size_type, MAX_DIMS> strides_;
    size_type                       rank_;
    MemoryLayout                    layout_;

    [[nodiscard]] size_type compute_offset(const std::array<size_type, MAX_DIMS>& indices) const;
};

} // namespace hpc::compute