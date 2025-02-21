#pragma once

#include "compute/cpp/tensor.hpp"
#include <memory>
#include <string>
#include <unordered_map>
#include <variant>

namespace hpc::compute {

enum class KernelType {
    Convolution2D
};

using ParamValue = std::variant<size_t, float, double, int, bool>;

class KernelDescriptor {
public:
    explicit KernelDescriptor(KernelType type) : type_(type) {}

    [[nodiscard]] KernelType type() const {
        return type_;
    }

    template <typename T>
    void set_param(std::string_view name, T value) {
        params_[std::string(name)] = value;
    }

    template <typename T>
    [[nodiscard]] T get_param(std::string_view name) const {
        if (auto it = params_.find(std::string(name)); it != params_.end()) {
            return std::get<T>(it->second);
        }
        throw std::runtime_error("Parameter not found: " + std::string(name));
    }

private:
    KernelType                                  type_;
    std::unordered_map<std::string, ParamValue> params_;
};

template <typename T>
class BaseKernel {
public:
    explicit BaseKernel(const KernelDescriptor& desc)
        : desc_(desc), layout_(MemoryLayout::RowMajor) {}

    virtual ~BaseKernel()                    = default;
    BaseKernel(const BaseKernel&)            = delete;
    BaseKernel& operator=(const BaseKernel&) = delete;
    BaseKernel(BaseKernel&&)                 = default;
    BaseKernel& operator=(BaseKernel&&)      = default;

    virtual void                    validate_input(const Tensor<T>& input) const = 0;
    [[nodiscard]] virtual Tensor<T> forward(const Tensor<T>& input) const        = 0;

    [[nodiscard]] const KernelDescriptor& descriptor() const {
        return desc_;
    }
    [[nodiscard]] virtual const Tensor<T>& weights() const                   = 0;
    virtual void                           load_weights(Tensor<T>&& weights) = 0;

    void set_layout(MemoryLayout layout) {
        layout_ = layout;
    }
    [[nodiscard]] MemoryLayout layout() const {
        return layout_;
    }

protected:
    KernelDescriptor desc_;
    MemoryLayout     layout_;
};

// Forward declaration of Conv2D.
template <typename T>
class Conv2D;

} // namespace hpc::compute