#pragma once

#include <array>
#include <concepts>
#include <cstddef>
#include <initializer_list>
#include <memory>
#include <span>
#include <type_traits>
#include <vector>

namespace hpc::compute {

template <typename T>
concept Numeric = std::is_arithmetic_v<T>;

enum class MemoryLayout {
    RowMajor,   // C-style, last dimension varies fastest
    ColumnMajor // Fortran-style, first dimension varies fastest
};

enum class MemoryOrder {
    Native,    // Use architecture's native alignment
    Aligned64, // Force 64-byte alignment for AVX-512
    Aligned32, // Force 32-byte alignment for AVX-256
    Packed     // No padding, tightly packed
};

// Forward declaration for TensorView
template <Numeric T>
class TensorView;

template <Numeric T>
class Tensor {
public:
    static constexpr std::size_t MAX_DIMS = 8;
    using value_type                      = T;
    using size_type                       = std::size_t;
    using pointer                         = T*;
    using const_pointer                   = const T*;
    using reference                       = T&;
    using const_reference                 = const T&;

    // Constructors
    Tensor() noexcept;

    explicit Tensor(std::initializer_list<size_type> dims,
                    MemoryLayout                     layout = MemoryLayout::RowMajor,
                    MemoryOrder                      order  = MemoryOrder::Aligned64);

    explicit Tensor(const std::vector<size_type>& dims,
                    MemoryLayout                  layout = MemoryLayout::RowMajor,
                    MemoryOrder                   order  = MemoryOrder::Aligned64);

    // Move operations
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    // Copy operations - disabled by default
    Tensor(const Tensor& other)            = delete;
    Tensor& operator=(const Tensor& other) = delete;

    // Destructor
    ~Tensor();

    // Data access
    pointer data() noexcept {
        return data_;
    }
    const_pointer data() const noexcept {
        return data_;
    }

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

    // View creation
    TensorView<T> view(std::initializer_list<std::pair<size_type, size_type>> ranges);
    TensorView<T> slice(size_type dim, size_type index);

    // Properties
    [[nodiscard]] size_type rank() const noexcept {
        return rank_;
    }
    [[nodiscard]] size_type size() const noexcept {
        return size_;
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
    [[nodiscard]] MemoryOrder order() const noexcept {
        return order_;
    }
    [[nodiscard]] size_type alignment() const noexcept;

    // Memory operations
    void               zero() noexcept;
    [[nodiscard]] bool is_contiguous() const noexcept;
    void               make_contiguous();

private:
    pointer                         data_;
    std::array<size_type, MAX_DIMS> dims_;
    std::array<size_type, MAX_DIMS> strides_;
    size_type                       rank_;
    size_type                       size_;
    MemoryLayout                    layout_;
    MemoryOrder                     order_;

    void                    allocate();
    void                    deallocate() noexcept;
    void                    compute_strides();
    [[nodiscard]] size_type compute_offset(const std::array<size_type, MAX_DIMS>& indices) const;
    static pointer          allocate_aligned(size_type size, size_type alignment);
    static void deallocate_aligned(pointer p, MemoryOrder order, size_type alignment) noexcept;
};

} // namespace hpc::compute