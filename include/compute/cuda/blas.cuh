#pragma once

#include "compute/cpp/tensor.hpp"
#include "compute/cuda/tensor.cuh"
#include "compute/cuda/wrapper.hpp"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <type_traits>
#include <vector>

namespace hpc::compute::cuda {

// Helper to get the cusparse data type based on C++ type
template <typename T>
cudaDataType cusparseDataType() {
    if (std::is_same_v<T, float>) {
        return CUDA_R_32F;
    } else if (std::is_same_v<T, double>) {
        return CUDA_R_64F;
    } else {
        throw std::runtime_error("Unsupported data type for cuSPARSE");
    }
}

enum class BlasImplementation {
    Custom,  // Custom CUDA kernel implementation
    CuBLAS,  // Using NVIDIA cuBLAS library
    CuSPARSE // Using NVIDIA cuSPARSE library
};

/**
 * CUDA-accelerated Basic Linear Algebra Subroutines (BLAS)
 */
template <typename T>
class BLAS {
public:
    // Select the best implementation based on operation and data characteristics
    static BlasImplementation select_implementation(bool is_sparse = false);

    /**
     * General Matrix Multiplication using CUDA: C = alpha * A * B + beta * C
     *
     * @param trans_a Whether to transpose A
     * @param trans_b Whether to transpose B
     * @param m Number of rows in A and C
     * @param n Number of columns in B and C
     * @param k Number of columns in A and rows in B
     * @param alpha Scaling factor for A * B
     * @param A Matrix A (m x k) on device memory
     * @param lda Leading dimension of A
     * @param B Matrix B (k x n) on device memory
     * @param ldb Leading dimension of B
     * @param beta Scaling factor for C
     * @param C Matrix C (m x n) on device memory
     * @param ldc Leading dimension of C
     * @param stream CUDA stream to use (optional)
     */
    static void gemm(bool         trans_a,
                     bool         trans_b,
                     size_t       m,
                     size_t       n,
                     size_t       k,
                     T            alpha,
                     const T*     A,
                     size_t       lda,
                     const T*     B,
                     size_t       ldb,
                     T            beta,
                     T*           C,
                     size_t       ldc,
                     cudaStream_t stream = nullptr);

    /**
     * TensorWrapper-based GEMM operation using CUDA
     */
    static void gemm(bool                    trans_a,
                     bool                    trans_b,
                     T                       alpha,
                     const TensorWrapper<T>& A,
                     const TensorWrapper<T>& B,
                     T                       beta,
                     TensorWrapper<T>&       C,
                     cudaStream_t            stream = nullptr);

    /**
     * Sparse General Matrix Multiplication using CUDA: C = alpha * A * B + beta * C
     * where A is a sparse matrix in CSR format
     *
     * @param trans_a Whether to transpose A
     * @param trans_b Whether to transpose B
     * @param m Number of rows in A and C
     * @param n Number of columns in B and C
     * @param k Number of columns in A and rows in B
     * @param alpha Scaling factor for A * B
     * @param csrRowPtrA CSR row pointers of A
     * @param csrColIndA CSR column indices of A
     * @param csrValA CSR values of A
     * @param nnzA Number of non-zero elements in A
     * @param B Matrix B (k x n) on device memory
     * @param ldb Leading dimension of B
     * @param beta Scaling factor for C
     * @param C Matrix C (m x n) on device memory
     * @param ldc Leading dimension of C
     * @param stream CUDA stream to use (optional)
     */
    static void sparse_gemm(bool         trans_a,
                            bool         trans_b,
                            size_t       m,
                            size_t       n,
                            size_t       k,
                            T            alpha,
                            const int*   csrRowPtrA,
                            const int*   csrColIndA,
                            const T*     csrValA,
                            int          nnzA,
                            const T*     B,
                            size_t       ldb,
                            T            beta,
                            T*           C,
                            size_t       ldc,
                            cudaStream_t stream = nullptr);

    /**
     * Image to Column transformation for efficient convolution using CUDA
     *
     * @param data Input image data on device memory
     * @param batch_size Batch size
     * @param channels Number of input channels
     * @param height Input height
     * @param width Input width
     * @param kernel_h Kernel height
     * @param kernel_w Kernel width
     * @param pad_h Padding height
     * @param pad_w Padding width
     * @param stride_h Stride height
     * @param stride_w Stride width
     * @param output_h Output height
     * @param output_w Output width
     * @param col_data Output column data on device memory
     * @param stream CUDA stream to use (optional)
     */
    static void im2col(const T*     data,
                       size_t       batch_size,
                       size_t       channels,
                       size_t       height,
                       size_t       width,
                       size_t       kernel_h,
                       size_t       kernel_w,
                       size_t       pad_h,
                       size_t       pad_w,
                       size_t       stride_h,
                       size_t       stride_w,
                       size_t       output_h,
                       size_t       output_w,
                       T*           col_data,
                       cudaStream_t stream = nullptr);

    /**
     * TensorWrapper-based im2col operation using CUDA
     */
    static void im2col(const TensorWrapper<T>& input,
                       size_t                  kernel_h,
                       size_t                  kernel_w,
                       size_t                  pad_h,
                       size_t                  pad_w,
                       size_t                  stride_h,
                       size_t                  stride_w,
                       TensorWrapper<T>&       col_output,
                       cudaStream_t            stream = nullptr);

    /**
     * Column to Image transformation (reverse of im2col) using CUDA
     *
     * @param col_data Input column data on device memory
     * @param batch_size Batch size
     * @param channels Number of input channels
     * @param height Input height
     * @param width Input width
     * @param kernel_h Kernel height
     * @param kernel_w Kernel width
     * @param pad_h Padding height
     * @param pad_w Padding width
     * @param stride_h Stride height
     * @param stride_w Stride width
     * @param output_h Output height
     * @param output_w Output width
     * @param data Output image data on device memory
     * @param stream CUDA stream to use (optional)
     */
    static void col2im(const T*     col_data,
                       size_t       batch_size,
                       size_t       channels,
                       size_t       height,
                       size_t       width,
                       size_t       kernel_h,
                       size_t       kernel_w,
                       size_t       pad_h,
                       size_t       pad_w,
                       size_t       stride_h,
                       size_t       stride_w,
                       size_t       output_h,
                       size_t       output_w,
                       T*           data,
                       cudaStream_t stream = nullptr);

    /**
     * TensorWrapper-based col2im operation using CUDA
     */
    static void col2im(const TensorWrapper<T>& col_input,
                       size_t                  channels,
                       size_t                  height,
                       size_t                  width,
                       size_t                  kernel_h,
                       size_t                  kernel_w,
                       size_t                  pad_h,
                       size_t                  pad_w,
                       size_t                  stride_h,
                       size_t                  stride_w,
                       TensorWrapper<T>&       output,
                       cudaStream_t            stream = nullptr);

    /**
     * Initialize CUDA BLAS resources (cuBLAS handle, etc.)
     * Call this before using any BLAS operations
     */
    static void initialize();

    /**
     * Release CUDA BLAS resources
     * Call this when done with BLAS operations
     */
    static void finalize();

private:
    // cuBLAS handle for dense operations
    static cublasHandle_t cublas_handle_;

    // cuSPARSE handle for sparse operations
    static cusparseHandle_t cusparse_handle_;

    // Flags to track initialization state
    static bool is_initialized_;

    // Implementation-specific GEMM functions
    static void gemm_custom(bool         trans_a,
                            bool         trans_b,
                            size_t       m,
                            size_t       n,
                            size_t       k,
                            T            alpha,
                            const T*     A,
                            size_t       lda,
                            const T*     B,
                            size_t       ldb,
                            T            beta,
                            T*           C,
                            size_t       ldc,
                            cudaStream_t stream);

    static void gemm_cublas(bool         trans_a,
                            bool         trans_b,
                            size_t       m,
                            size_t       n,
                            size_t       k,
                            T            alpha,
                            const T*     A,
                            size_t       lda,
                            const T*     B,
                            size_t       ldb,
                            T            beta,
                            T*           C,
                            size_t       ldc,
                            cudaStream_t stream);

    static void gemm_cusparse(bool         trans_a,
                              bool         trans_b,
                              size_t       m,
                              size_t       n,
                              size_t       k,
                              T            alpha,
                              const int*   csrRowPtrA,
                              const int*   csrColIndA,
                              const T*     csrValA,
                              int          nnzA,
                              const T*     B,
                              size_t       ldb,
                              T            beta,
                              T*           C,
                              size_t       ldc,
                              cudaStream_t stream);

    // Implementation-specific im2col functions
    static void im2col_custom(const T*     data,
                              size_t       batch_size,
                              size_t       channels,
                              size_t       height,
                              size_t       width,
                              size_t       kernel_h,
                              size_t       kernel_w,
                              size_t       pad_h,
                              size_t       pad_w,
                              size_t       stride_h,
                              size_t       stride_w,
                              size_t       output_h,
                              size_t       output_w,
                              T*           col_data,
                              cudaStream_t stream);

    // Implementation-specific col2im functions
    static void col2im_custom(const T*     col_data,
                              size_t       batch_size,
                              size_t       channels,
                              size_t       height,
                              size_t       width,
                              size_t       kernel_h,
                              size_t       kernel_w,
                              size_t       pad_h,
                              size_t       pad_w,
                              size_t       stride_h,
                              size_t       stride_w,
                              size_t       output_h,
                              size_t       output_w,
                              T*           data,
                              cudaStream_t stream);

    // Helper methods for CUDA error handling
    static void check_cuda_error(cudaError_t error, const char* func, const char* file, int line);
    static void check_cublas_error(cublasStatus_t status,
                                   const char*    func,
                                   const char*    file,
                                   int            line);
    static void check_cusparse_error(cusparseStatus_t status,
                                     const char*      func,
                                     const char*      file,
                                     int              line);
};

// Initialize static members
template <typename T>
cublasHandle_t BLAS<T>::cublas_handle_ = nullptr;

template <typename T>
cusparseHandle_t BLAS<T>::cusparse_handle_ = nullptr;

template <typename T>
bool BLAS<T>::is_initialized_ = false;

// Define macros for error checking
#define CHECK_CUDA_ERROR(err) check_cuda_error(err, __FUNCTION__, __FILE__, __LINE__)
#define CHECK_CUBLAS_ERROR(err) check_cublas_error(err, __FUNCTION__, __FILE__, __LINE__)
#define CHECK_CUSPARSE_ERROR(err) check_cusparse_error(err, __FUNCTION__, __FILE__, __LINE__)

} // namespace hpc::compute::cuda