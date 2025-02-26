#include "compute/cuda/blas.cuh"
#include <algorithm>
#include <iostream>
#include <stdexcept>

// Define block size for GEMM kernel
#define GEMM_BLOCK_SIZE 32

namespace hpc::compute::cuda {

// Error handling functions
template <typename T>
void BLAS<T>::check_cuda_error(cudaError_t error, const char* func, const char* file, int line) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in " << func << " at " << file << ":" << line << ": "
                  << cudaGetErrorString(error) << " (" << error << ")" << std::endl;
        throw std::runtime_error("CUDA error");
    }
}

template <typename T>
void BLAS<T>::check_cublas_error(cublasStatus_t status,
                                 const char*    func,
                                 const char*    file,
                                 int            line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS error in " << func << " at " << file << ":" << line << ": " << status
                  << std::endl;
        throw std::runtime_error("cuBLAS error");
    }
}

template <typename T>
void BLAS<T>::check_cusparse_error(cusparseStatus_t status,
                                   const char*      func,
                                   const char*      file,
                                   int              line) {
    if (status != CUSPARSE_STATUS_SUCCESS) {
        std::cerr << "cuSPARSE error in " << func << " at " << file << ":" << line << ": " << status
                  << std::endl;
        throw std::runtime_error("cuSPARSE error");
    }
}

// Initialize CUDA BLAS resources
template <typename T>
void BLAS<T>::initialize() {
    if (!is_initialized_) {
        // Create cuBLAS handle
        CHECK_CUBLAS_ERROR(cublasCreate(&cublas_handle_));

        // Create cuSPARSE handle
        CHECK_CUSPARSE_ERROR(cusparseCreate(&cusparse_handle_));

        is_initialized_ = true;
    }
}

// Release CUDA BLAS resources
template <typename T>
void BLAS<T>::finalize() {
    if (is_initialized_) {
        if (cublas_handle_ != nullptr) {
            CHECK_CUBLAS_ERROR(cublasDestroy(cublas_handle_));
            cublas_handle_ = nullptr;
        }

        if (cusparse_handle_ != nullptr) {
            CHECK_CUSPARSE_ERROR(cusparseDestroy(cusparse_handle_));
            cusparse_handle_ = nullptr;
        }

        is_initialized_ = false;
    }
}

// Select the best implementation based on data characteristics
template <typename T>
BlasImplementation BLAS<T>::select_implementation(bool is_sparse) {
    if (is_sparse) {
        return BlasImplementation::CuSPARSE;
    } else {
        // For most dense operations, cuBLAS is faster than custom implementations
        return BlasImplementation::CuBLAS;
    }
}

// GEMM implementation dispatching
template <typename T>
void BLAS<T>::gemm(bool         trans_a,
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
                   cudaStream_t stream) {
    // Ensure resources are initialized
    if (!is_initialized_) {
        initialize();
    }

    // By default, use cuBLAS
    gemm_cublas(trans_a, trans_b, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, stream);
}

// TensorWrapper-based GEMM
template <typename T>
void BLAS<T>::gemm(bool                    trans_a,
                   bool                    trans_b,
                   T                       alpha,
                   const TensorWrapper<T>& A,
                   const TensorWrapper<T>& B,
                   T                       beta,
                   TensorWrapper<T>&       C,
                   cudaStream_t            stream) {
    // Validate tensor shapes based on transposition flags
    const auto& A_shape = A.tensor_data().dims;
    const auto& B_shape = B.tensor_data().dims;
    const auto& C_shape = C.tensor_data().dims;

    size_t A_rows = trans_a ? A_shape[1] : A_shape[0];
    size_t A_cols = trans_a ? A_shape[0] : A_shape[1];
    size_t B_rows = trans_b ? B_shape[1] : B_shape[0];
    size_t B_cols = trans_b ? B_shape[0] : B_shape[1];

    // Check dimensions compatibility
    if (A_cols != B_rows) {
        throw std::invalid_argument("Incompatible matrix dimensions for GEMM");
    }

    if (A_rows != C_shape[0] || B_cols != C_shape[1]) {
        throw std::invalid_argument("Output matrix has incorrect dimensions");
    }

    // Call raw GEMM function
    gemm(trans_a,
         trans_b,
         A_rows,
         B_cols,
         A_cols,
         alpha,
         A.tensor_data().data,
         A_shape[1], // lda is stride for first dimension
         B.tensor_data().data,
         B_shape[1], // ldb is stride for first dimension
         beta,
         C.tensor_data().data,
         C_shape[1], // ldc is stride for first dimension
         stream);
}

// Sparse GEMM implementation
template <typename T>
void BLAS<T>::sparse_gemm(bool         trans_a,
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
                          cudaStream_t stream) {
    // Ensure resources are initialized
    if (!is_initialized_) {
        initialize();
    }

    // Use cuSPARSE for sparse matrix operations
    gemm_cusparse(trans_a,
                  trans_b,
                  m,
                  n,
                  k,
                  alpha,
                  csrRowPtrA,
                  csrColIndA,
                  csrValA,
                  nnzA,
                  B,
                  ldb,
                  beta,
                  C,
                  ldc,
                  stream);
}

// cuBLAS GEMM implementation
template <typename T>
void BLAS<T>::gemm_cublas(bool         trans_a,
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
                          cudaStream_t stream) {
    // Set stream for cuBLAS operations
    if (stream != nullptr) {
        CHECK_CUBLAS_ERROR(cublasSetStream(cublas_handle_, stream));
    }

    // cuBLAS uses column-major ordering, so we need to compute C^T = B^T * A^T
    // which is equivalent to C = A * B in row-major ordering
    cublasOperation_t trans_a_op = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t trans_b_op = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;

    // Specialize for float and double types
    if constexpr (std::is_same_v<T, float>) {
        CHECK_CUBLAS_ERROR(cublasSgemm(cublas_handle_,
                                       trans_b_op,
                                       trans_a_op,
                                       n,
                                       m,
                                       k,
                                       &alpha,
                                       B,
                                       ldb,
                                       A,
                                       lda,
                                       &beta,
                                       C,
                                       ldc));
    } else if constexpr (std::is_same_v<T, double>) {
        CHECK_CUBLAS_ERROR(cublasDgemm(cublas_handle_,
                                       trans_b_op,
                                       trans_a_op,
                                       n,
                                       m,
                                       k,
                                       &alpha,
                                       B,
                                       ldb,
                                       A,
                                       lda,
                                       &beta,
                                       C,
                                       ldc));
    } else {
        throw std::runtime_error("Unsupported data type for cuBLAS GEMM");
    }
}

// cuSPARSE GEMM implementation (for sparse matrices)
template <typename T>
void BLAS<T>::gemm_cusparse(bool         trans_a,
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
                            cudaStream_t stream) {
    // Set stream for cuSPARSE operations
    if (stream != nullptr) {
        CHECK_CUSPARSE_ERROR(cusparseSetStream(cusparse_handle_, stream));
    }

    // Create matrix descriptors for the new generic API
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    void*                dBuffer    = nullptr;
    size_t               bufferSize = 0;

    // Operation types
    cusparseOperation_t opA =
        trans_a ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t opB =
        trans_b ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;

    // Get the actual dimensions based on transpose flags
    int A_rows = trans_a ? k : m;
    int A_cols = trans_a ? m : k;
    int B_rows = trans_b ? n : k;
    int B_cols = trans_b ? k : n;

    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE_ERROR(cusparseCreateCsr(&matA,  // matrix descriptor
                                           A_rows, // number of rows
                                           A_cols, // number of columns
                                           nnzA,   // number of non-zero elements
                                           const_cast<int*>(csrRowPtrA), // row offsets
                                           const_cast<int*>(csrColIndA), // column indices
                                           const_cast<T*>(csrValA),      // values
                                           CUSPARSE_INDEX_32I, // index type for row offsets
                                           CUSPARSE_INDEX_32I, // index type for column indices
                                           CUSPARSE_INDEX_BASE_ZERO, // base index (0 or 1)
                                           cusparseDataType<T>()     // data type
                                           ));

    // Create dense matrix B
    CHECK_CUSPARSE_ERROR(
        cusparseCreateDnMat(&matB,                 // matrix descriptor
                            B_rows,                // number of rows
                            B_cols,                // number of columns
                            ldb,                   // leading dimension
                            const_cast<T*>(B),     // values
                            cusparseDataType<T>(), // data type
                            CUSPARSE_ORDER_ROW     // memory layout (row or column major)
                            ));

    // Create dense matrix C
    CHECK_CUSPARSE_ERROR(
        cusparseCreateDnMat(&matC,                 // matrix descriptor
                            m,                     // number of rows
                            n,                     // number of columns
                            ldc,                   // leading dimension
                            C,                     // values
                            cusparseDataType<T>(), // data type
                            CUSPARSE_ORDER_ROW     // memory layout (row or column major)
                            ));

    // Get buffer size required for SpMM
    CHECK_CUSPARSE_ERROR(cusparseSpMM_bufferSize(cusparse_handle_,
                                                 opA,
                                                 opB,
                                                 &alpha,
                                                 matA,
                                                 matB,
                                                 &beta,
                                                 matC,
                                                 cusparseDataType<T>(),
                                                 CUSPARSE_SPMM_ALG_DEFAULT,
                                                 &bufferSize));

    // Allocate external buffer if needed
    if (bufferSize > 0) {
        CHECK_CUDA_ERROR(cudaMalloc(&dBuffer, bufferSize));
    }

    // Execute SpMM operation
    CHECK_CUSPARSE_ERROR(cusparseSpMM(cusparse_handle_,
                                      opA,
                                      opB,
                                      &alpha,
                                      matA,
                                      matB,
                                      &beta,
                                      matC,
                                      cusparseDataType<T>(),
                                      CUSPARSE_SPMM_ALG_DEFAULT,
                                      dBuffer));

    // Free resources
    if (dBuffer) {
        CHECK_CUDA_ERROR(cudaFree(dBuffer));
    }
    CHECK_CUSPARSE_ERROR(cusparseDestroySpMat(matA));
    CHECK_CUSPARSE_ERROR(cusparseDestroyDnMat(matB));
    CHECK_CUSPARSE_ERROR(cusparseDestroyDnMat(matC));
}

// Custom GEMM CUDA kernel for matrix multiplication C = alpha*A*B + beta*C
template <typename T>
__global__ void gemm_kernel(bool     trans_a,
                            bool     trans_b,
                            int      m,
                            int      n,
                            int      k,
                            T        alpha,
                            const T* A,
                            int      lda,
                            const T* B,
                            int      ldb,
                            T        beta,
                            T*       C,
                            int      ldc) {
    // Shared memory for tiles
    __shared__ T shared_A[GEMM_BLOCK_SIZE][GEMM_BLOCK_SIZE];
    __shared__ T shared_B[GEMM_BLOCK_SIZE][GEMM_BLOCK_SIZE];

    // Block and thread indices
    int block_row  = blockIdx.y;
    int block_col  = blockIdx.x;
    int thread_row = threadIdx.y;
    int thread_col = threadIdx.x;

    // Calculate global row and column indices
    int row = block_row * GEMM_BLOCK_SIZE + thread_row;
    int col = block_col * GEMM_BLOCK_SIZE + thread_col;

    // Accumulator for dot product
    T sum = 0;

    // For each tile
    for (int tile = 0; tile < (k + GEMM_BLOCK_SIZE - 1) / GEMM_BLOCK_SIZE; ++tile) {
        // Load the shared memory tiles
        if (trans_a) {
            int a_col = tile * GEMM_BLOCK_SIZE + thread_col;
            if (row < m && a_col < k) {
                shared_A[thread_row][thread_col] = A[a_col * lda + row];
            } else {
                shared_A[thread_row][thread_col] = 0;
            }
        } else {
            int a_col = tile * GEMM_BLOCK_SIZE + thread_col;
            if (row < m && a_col < k) {
                shared_A[thread_row][thread_col] = A[row * lda + a_col];
            } else {
                shared_A[thread_row][thread_col] = 0;
            }
        }

        if (trans_b) {
            int b_row = tile * GEMM_BLOCK_SIZE + thread_row;
            if (b_row < k && col < n) {
                shared_B[thread_row][thread_col] = B[col * ldb + b_row];
            } else {
                shared_B[thread_row][thread_col] = 0;
            }
        } else {
            int b_row = tile * GEMM_BLOCK_SIZE + thread_row;
            if (b_row < k && col < n) {
                shared_B[thread_row][thread_col] = B[b_row * ldb + col];
            } else {
                shared_B[thread_row][thread_col] = 0;
            }
        }

        __syncthreads();

// Multiply the two matrices together
#pragma unroll
        for (int e = 0; e < GEMM_BLOCK_SIZE; ++e) {
            sum += shared_A[thread_row][e] * shared_B[e][thread_col];
        }

        __syncthreads();
    }

    // Write the result to global memory
    if (row < m && col < n) {
        if (beta == 0) {
            C[row * ldc + col] = alpha * sum;
        } else {
            C[row * ldc + col] = alpha * sum + beta * C[row * ldc + col];
        }
    }
}

template <typename T>
void BLAS<T>::gemm_custom(bool         trans_a,
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
                          cudaStream_t stream) {
    // Define block and grid dimensions
    dim3 blockDim(GEMM_BLOCK_SIZE, GEMM_BLOCK_SIZE);
    dim3 gridDim((n + GEMM_BLOCK_SIZE - 1) / GEMM_BLOCK_SIZE,
                 (m + GEMM_BLOCK_SIZE - 1) / GEMM_BLOCK_SIZE);

    // Launch kernel
    gemm_kernel<<<gridDim, blockDim, 0, stream>>>(
        trans_a, trans_b, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);

    // Check for kernel launch errors
    CHECK_CUDA_ERROR(cudaGetLastError());
}

// CUDA kernel for im2col operation
template <typename T>
__global__ void im2col_kernel(const T* data_im,
                              int      channels,
                              int      height,
                              int      width,
                              int      kernel_h,
                              int      kernel_w,
                              int      pad_h,
                              int      pad_w,
                              int      stride_h,
                              int      stride_w,
                              int      output_h,
                              int      output_w,
                              T*       data_col) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < channels * kernel_h * kernel_w * output_h * output_w) {
        int w_out = index % output_w;
        index /= output_w;
        int h_out = index % output_h;
        index /= output_h;
        int w_in = index % kernel_w;
        index /= kernel_w;
        int h_in = index % kernel_h;
        index /= kernel_h;
        int c_in = index;

        int h_im = h_out * stride_h - pad_h + h_in;
        int w_im = w_out * stride_w - pad_w + w_in;

        data_col[(((c_in * kernel_h + h_in) * kernel_w + w_in) * output_h + h_out) * output_w +
                 w_out] = (h_im >= 0 && h_im < height && w_im >= 0 && w_im < width)
                              ? data_im[(c_in * height + h_im) * width + w_im]
                              : T(0);
    }
}

// im2col operation
template <typename T>
void BLAS<T>::im2col(const T*     data,
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
                     cudaStream_t stream) {
    // We'll use the custom implementation for im2col
    im2col_custom(data,
                  batch_size,
                  channels,
                  height,
                  width,
                  kernel_h,
                  kernel_w,
                  pad_h,
                  pad_w,
                  stride_h,
                  stride_w,
                  output_h,
                  output_w,
                  col_data,
                  stream);
}

// TensorWrapper-based im2col
template <typename T>
void BLAS<T>::im2col(const TensorWrapper<T>& input,
                     size_t                  kernel_h,
                     size_t                  kernel_w,
                     size_t                  pad_h,
                     size_t                  pad_w,
                     size_t                  stride_h,
                     size_t                  stride_w,
                     TensorWrapper<T>&       col_output,
                     cudaStream_t            stream) {
    // Validate input tensor dimensions (must be a 4D tensor [batch, channels, height, width])
    const auto& input_shape = input.tensor_data().dims;
    if (input.rank() != 4) {
        throw std::invalid_argument("Input tensor must be 4D [batch, channels, height, width]");
    }

    size_t batch_size = input_shape[0];
    size_t channels   = input_shape[1];
    size_t height     = input_shape[2];
    size_t width      = input_shape[3];

    // Calculate output dimensions
    size_t output_h = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    size_t output_w = (width + 2 * pad_w - kernel_w) / stride_w + 1;

    // Ensure col_output has the correct shape
    // The output will have shape [batch_size, channels * kernel_h * kernel_w, output_h * output_w]
    // or effectively [batch_size, channels * kernel_h * kernel_w, output_h, output_w]
    const auto& col_shape     = col_output.tensor_data().dims;
    size_t      expected_size = batch_size * channels * kernel_h * kernel_w * output_h * output_w;

    if (col_output.size() != expected_size) {
        throw std::invalid_argument("Output tensor has incorrect size for im2col operation");
    }

    // Process each batch separately
    for (size_t b = 0; b < batch_size; ++b) {
        // Calculate pointers for current batch
        const T* input_batch = input.tensor_data().data + b * channels * height * width;
        T*       col_batch   = col_output.tensor_data().data +
                       b * channels * kernel_h * kernel_w * output_h * output_w;

        // Call the raw im2col function
        im2col(input_batch,
               1,
               channels,
               height,
               width,
               kernel_h,
               kernel_w,
               pad_h,
               pad_w,
               stride_h,
               stride_w,
               output_h,
               output_w,
               col_batch,
               stream);
    }
}

// Custom im2col implementation
template <typename T>
void BLAS<T>::im2col_custom(const T*     data,
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
                            cudaStream_t stream) {
    // Calculate kernel launch parameters
    size_t        total_threads     = channels * kernel_h * kernel_w * output_h * output_w;
    constexpr int threads_per_block = 256;
    int           num_blocks        = (total_threads + threads_per_block - 1) / threads_per_block;

    // Launch kernel for each batch
    for (size_t b = 0; b < batch_size; ++b) {
        const T* input_batch = data + b * channels * height * width;
        T*       col_batch   = col_data + b * channels * kernel_h * kernel_w * output_h * output_w;

        im2col_kernel<<<num_blocks, threads_per_block, 0, stream>>>(input_batch,
                                                                    channels,
                                                                    height,
                                                                    width,
                                                                    kernel_h,
                                                                    kernel_w,
                                                                    pad_h,
                                                                    pad_w,
                                                                    stride_h,
                                                                    stride_w,
                                                                    output_h,
                                                                    output_w,
                                                                    col_batch);

        // Check for CUDA errors
        CHECK_CUDA_ERROR(cudaGetLastError());
    }
}

// CUDA kernel for col2im operation
template <typename T>
__global__ void col2im_kernel(const T* data_col,
                              int      channels,
                              int      height,
                              int      width,
                              int      kernel_h,
                              int      kernel_w,
                              int      pad_h,
                              int      pad_w,
                              int      stride_h,
                              int      stride_w,
                              int      output_h,
                              int      output_w,
                              T*       data_im) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < channels * height * width) {
        T   val = 0;
        int w   = index % width;
        index /= width;
        int h = index % height;
        index /= height;
        int c = index;

        // Compute the region in the column buffer that contributes to this pixel
        int kernel_extent_h = kernel_h - 1;
        int kernel_extent_w = kernel_w - 1;
        int h_start         = max(0, (h - kernel_extent_h - pad_h + stride_h - 1) / stride_h + 1);
        int h_end           = min(output_h, (h + pad_h) / stride_h + 1);
        int w_start         = max(0, (w - kernel_extent_w - pad_w + stride_w - 1) / stride_w + 1);
        int w_end           = min(output_w, (w + pad_w) / stride_w + 1);

        // Count the number of contributing patches
        int count = 0;

        // Accumulate values from all contributing patches
        for (int oh = h_start; oh < h_end; ++oh) {
            for (int ow = w_start; ow < w_end; ++ow) {
                int kh = h - oh * stride_h + pad_h;
                int kw = w - ow * stride_w + pad_w;

                if (kh >= 0 && kh < kernel_h && kw >= 0 && kw < kernel_w) {
                    val += data_col[(((c * kernel_h + kh) * kernel_w + kw) * output_h + oh) *
                                        output_w +
                                    ow];
                    count++;
                }
            }
        }

        // Normalize by dividing by the count of contributing patches
        data_im[(c * height + h) * width + w] = (count > 0) ? val / count : 0;
    }
}

// col2im operation
template <typename T>
void BLAS<T>::col2im(const T*     col_data,
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
                     cudaStream_t stream) {
    // We'll use the custom implementation for col2im
    col2im_custom(col_data,
                  batch_size,
                  channels,
                  height,
                  width,
                  kernel_h,
                  kernel_w,
                  pad_h,
                  pad_w,
                  stride_h,
                  stride_w,
                  output_h,
                  output_w,
                  data,
                  stream);
}

// TensorWrapper-based col2im
template <typename T>
void BLAS<T>::col2im(const TensorWrapper<T>& col_input,
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
                     cudaStream_t            stream) {
    // Calculate output dimensions
    size_t output_h = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    size_t output_w = (width + 2 * pad_w - kernel_w) / stride_w + 1;

    // Validate col_input tensor dimensions
    const auto& col_shape  = col_input.tensor_data().dims;
    size_t      batch_size = output.tensor_data().dims[0];

    // Verify output tensor has the correct shape [batch_size, channels, height, width]
    const auto& output_shape = output.tensor_data().dims;
    if (output.rank() != 4 || output_shape[0] != batch_size || output_shape[1] != channels ||
        output_shape[2] != height || output_shape[3] != width) {
        throw std::invalid_argument("Output tensor has incorrect shape for col2im operation");
    }

    // Process each batch separately
    for (size_t b = 0; b < batch_size; ++b) {
        // Calculate pointers for current batch
        const T* col_batch =
            col_input.tensor_data().data + b * channels * kernel_h * kernel_w * output_h * output_w;
        T* output_batch = output.tensor_data().data + b * channels * height * width;

        // Call the raw col2im function
        col2im(col_batch,
               1,
               channels,
               height,
               width,
               kernel_h,
               kernel_w,
               pad_h,
               pad_w,
               stride_h,
               stride_w,
               output_h,
               output_w,
               output_batch,
               stream);
    }
}

// Custom col2im implementation
template <typename T>
void BLAS<T>::col2im_custom(const T*     col_data,
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
                            cudaStream_t stream) {
    // First, zero out the output data
    size_t img_size = channels * height * width;
    CHECK_CUDA_ERROR(cudaMemsetAsync(data, 0, batch_size * img_size * sizeof(T), stream));

    // Calculate kernel launch parameters
    constexpr int threads_per_block = 256;
    int           num_blocks        = (img_size + threads_per_block - 1) / threads_per_block;

    // Launch kernel for each batch
    for (size_t b = 0; b < batch_size; ++b) {
        const T* col_batch    = col_data + b * channels * kernel_h * kernel_w * output_h * output_w;
        T*       output_batch = data + b * img_size;

        col2im_kernel<<<num_blocks, threads_per_block, 0, stream>>>(col_batch,
                                                                    channels,
                                                                    height,
                                                                    width,
                                                                    kernel_h,
                                                                    kernel_w,
                                                                    pad_h,
                                                                    pad_w,
                                                                    stride_h,
                                                                    stride_w,
                                                                    output_h,
                                                                    output_w,
                                                                    output_batch);

        // Check for CUDA errors
        CHECK_CUDA_ERROR(cudaGetLastError());
    }
}

// Explicit template instantiations
template class BLAS<float>;
template class BLAS<double>;

} // namespace hpc::compute::cuda