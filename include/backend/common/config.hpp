#pragma once

#include "types.hpp"
#include <optional>

namespace kernel_compute {

struct KernelConfig {
    std::size_t batch_size{1};
    std::size_t channels{3};
    std::size_t height{224};
    std::size_t width{224};
    std::size_t kernel_size{3};
    std::size_t stride{1};
    std::size_t padding{1};
    KernelType  type{KernelType::Dense};

    // Sparse-specific parameters
    std::optional<std::size_t>          nnz;
    std::optional<std::vector<int32_t>> row_ptr;
    std::optional<std::vector<int32_t>> col_ind;

    // Validation
    bool validate() const {
        bool valid = batch_size > 0 && channels > 0 && height > 0 && width > 0 && kernel_size > 0 &&
                     stride > 0;

        if (type == KernelType::Sparse) {
            valid &=
                nnz.has_value() && row_ptr.has_value() && col_ind.has_value() && nnz.value() > 0;
        }

        return valid;
    }
};

struct ComputeConfig {
    DeviceType device{DeviceType::CPU};
    DataType   data_type{DataType::Float32};
    bool       enable_profiling{false};

    // Device-specific settings
    struct CudaSettings {
        int         device_id{0};
        std::size_t max_threads_per_block{256};
    } cuda;

    struct CpuSettings {
        std::size_t num_threads{1};
        bool        use_avx512{true};
    } cpu;
};

} // namespace kernel_compute