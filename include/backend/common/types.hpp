#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace kernel_compute {

enum class DeviceType {
    CPU,
    CUDA
};

enum class DataType {
    Float32,
    Float64,
    Int32,
    Int64
};

enum class KernelType {
    Dense,
    Sparse
};

// Forward declarations
class Tensor;
class Engine;
class KernelBase;

using Shape     = std::vector<std::size_t>;
using float32_t = float;
using float64_t = double;

} // namespace kernel_compute