#pragma once

#include "compute/cpp/tensor.hpp"
#include <cstddef>
#include <type_traits>
#include <vector>

// Forward declaration for CPU features since it's already implemented elsewhere
namespace hpc::compute {
struct CPUFeatures;
}

namespace hpc::compute {

enum class BlasImplementation {
    Basic,
    SSE42,
    AVX,
    AVX2,
    FMA,
    AVX512
};

/**
 * Basic Linear Algebra Subroutines (BLAS) for high-performance computing
 */
template <typename T>
class BLAS {
public:
    // Select the best implementation based on available CPU features and data type
    static BlasImplementation select_implementation();

    /**
     * General Matrix Multiplication: C = alpha * A * B + beta * C
     * with automatic SIMD optimization based on CPU features
     *
     * @param m Number of rows in A and C
     * @param n Number of columns in B and C
     * @param k Number of columns in A and rows in B
     * @param alpha Scaling factor for A * B
     * @param A Matrix A (m x k)
     * @param lda Leading dimension of A
     * @param B Matrix B (k x n)
     * @param ldb Leading dimension of B
     * @param beta Scaling factor for C
     * @param C Matrix C (m x n)
     * @param ldc Leading dimension of C
     */
    static void gemm(bool     trans_a,
                     bool     trans_b,
                     size_t   m,
                     size_t   n,
                     size_t   k,
                     T        alpha,
                     const T* A,
                     size_t   lda,
                     const T* B,
                     size_t   ldb,
                     T        beta,
                     T*       C,
                     size_t   ldc);

    /**
     * Tensor-based GEMM operation optimized for Tensor layout and alignment
     */
    static void gemm(bool             trans_a,
                     bool             trans_b,
                     T                alpha,
                     const Tensor<T>& A,
                     const Tensor<T>& B,
                     T                beta,
                     Tensor<T>&       C);

    /**
     * Image to Column transformation for efficient convolution
     * with automatic SIMD optimization based on CPU features
     *
     * @param data Input image data
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
     * @param col_data Output column data
     */
    static void im2col(const T* data,
                       size_t   batch_size,
                       size_t   channels,
                       size_t   height,
                       size_t   width,
                       size_t   kernel_h,
                       size_t   kernel_w,
                       size_t   pad_h,
                       size_t   pad_w,
                       size_t   stride_h,
                       size_t   stride_w,
                       size_t   output_h,
                       size_t   output_w,
                       T*       col_data);

    /**
     * Tensor-based im2col operation optimized for Tensor layout and alignment
     */
    static void im2col(const Tensor<T>& input,
                       size_t           kernel_h,
                       size_t           kernel_w,
                       size_t           pad_h,
                       size_t           pad_w,
                       size_t           stride_h,
                       size_t           stride_w,
                       Tensor<T>&       col_output);

    /**
     * Column to Image transformation (reverse of im2col)
     * with automatic SIMD optimization based on CPU features
     *
     * @param col_data Input column data
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
     * @param data Output image data
     */
    static void col2im(const T* col_data,
                       size_t   batch_size,
                       size_t   channels,
                       size_t   height,
                       size_t   width,
                       size_t   kernel_h,
                       size_t   kernel_w,
                       size_t   pad_h,
                       size_t   pad_w,
                       size_t   stride_h,
                       size_t   stride_w,
                       size_t   output_h,
                       size_t   output_w,
                       T*       data);

    /**
     * Tensor-based col2im operation optimized for Tensor layout and alignment
     */
    static void col2im(const Tensor<T>& col_input,
                       size_t           channels,
                       size_t           height,
                       size_t           width,
                       size_t           kernel_h,
                       size_t           kernel_w,
                       size_t           pad_h,
                       size_t           pad_w,
                       size_t           stride_h,
                       size_t           stride_w,
                       Tensor<T>&       output);

private:
    // Implementation-specific GEMM
    static void gemm_basic(bool     trans_a,
                           bool     trans_b,
                           size_t   m,
                           size_t   n,
                           size_t   k,
                           T        alpha,
                           const T* A,
                           size_t   lda,
                           const T* B,
                           size_t   ldb,
                           T        beta,
                           T*       C,
                           size_t   ldc);

    static void gemm_sse42(bool     trans_a,
                           bool     trans_b,
                           size_t   m,
                           size_t   n,
                           size_t   k,
                           T        alpha,
                           const T* A,
                           size_t   lda,
                           const T* B,
                           size_t   ldb,
                           T        beta,
                           T*       C,
                           size_t   ldc);

    static void gemm_avx(bool     trans_a,
                         bool     trans_b,
                         size_t   m,
                         size_t   n,
                         size_t   k,
                         T        alpha,
                         const T* A,
                         size_t   lda,
                         const T* B,
                         size_t   ldb,
                         T        beta,
                         T*       C,
                         size_t   ldc);

    static void gemm_avx2(bool     trans_a,
                          bool     trans_b,
                          size_t   m,
                          size_t   n,
                          size_t   k,
                          T        alpha,
                          const T* A,
                          size_t   lda,
                          const T* B,
                          size_t   ldb,
                          T        beta,
                          T*       C,
                          size_t   ldc);

    static void gemm_fma(bool     trans_a,
                         bool     trans_b,
                         size_t   m,
                         size_t   n,
                         size_t   k,
                         T        alpha,
                         const T* A,
                         size_t   lda,
                         const T* B,
                         size_t   ldb,
                         T        beta,
                         T*       C,
                         size_t   ldc);

    static void gemm_avx512(bool     trans_a,
                            bool     trans_b,
                            size_t   m,
                            size_t   n,
                            size_t   k,
                            T        alpha,
                            const T* A,
                            size_t   lda,
                            const T* B,
                            size_t   ldb,
                            T        beta,
                            T*       C,
                            size_t   ldc);

    // Implementation-specific im2col
    static void im2col_basic(const T* data,
                             size_t   batch_size,
                             size_t   channels,
                             size_t   height,
                             size_t   width,
                             size_t   kernel_h,
                             size_t   kernel_w,
                             size_t   pad_h,
                             size_t   pad_w,
                             size_t   stride_h,
                             size_t   stride_w,
                             size_t   output_h,
                             size_t   output_w,
                             T*       col_data);

    static void im2col_simd(const T* data,
                            size_t   batch_size,
                            size_t   channels,
                            size_t   height,
                            size_t   width,
                            size_t   kernel_h,
                            size_t   kernel_w,
                            size_t   pad_h,
                            size_t   pad_w,
                            size_t   stride_h,
                            size_t   stride_w,
                            size_t   output_h,
                            size_t   output_w,
                            T*       col_data);

    // Implementation-specific col2im
    static void col2im_basic(const T* col_data,
                             size_t   batch_size,
                             size_t   channels,
                             size_t   height,
                             size_t   width,
                             size_t   kernel_h,
                             size_t   kernel_w,
                             size_t   pad_h,
                             size_t   pad_w,
                             size_t   stride_h,
                             size_t   stride_w,
                             size_t   output_h,
                             size_t   output_w,
                             T*       data);

    static void col2im_simd(const T* col_data,
                            size_t   batch_size,
                            size_t   channels,
                            size_t   height,
                            size_t   width,
                            size_t   kernel_h,
                            size_t   kernel_w,
                            size_t   pad_h,
                            size_t   pad_w,
                            size_t   stride_h,
                            size_t   stride_w,
                            size_t   output_h,
                            size_t   output_w,
                            T*       data);

    // Helper methods for tensor operations
    static void compute_im2col_indices(size_t               height,
                                       size_t               width,
                                       size_t               kernel_h,
                                       size_t               kernel_w,
                                       size_t               pad_h,
                                       size_t               pad_w,
                                       size_t               stride_h,
                                       size_t               stride_w,
                                       size_t               output_h,
                                       size_t               output_w,
                                       std::vector<size_t>& kernel_indices,
                                       std::vector<size_t>& input_indices);
};

} // namespace hpc::compute