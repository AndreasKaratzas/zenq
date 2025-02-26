#include "compute/cpp/blas.hpp"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>
#include <omp.h>

// Include SIMD intrinsics based on platform
#ifdef _MSC_VER
    #include <intrin.h>
#else
    #include <x86intrin.h>
#endif

#include "common/logging.hpp"             // For logging
#include "compute/cpp/kernels/conv2d.hpp" // For CPUFeatures definition

namespace hpc::compute {

// Create our own instance of CPU features for BLAS
static const CPUFeatures blas_cpu_features = CPUFeatures::detect();

// Extern declaration of the utility function from conv2d.cpp
extern void logCPUFeatures(const std::string& component);

// Log available features at initialization
static bool logged_blas_features = []() {
    logCPUFeatures("BLAS");
    return true;
}();

// Implementation Selector
template <typename T>
BlasImplementation BLAS<T>::select_implementation() {
    std::string        type_name = typeid(T).name();
    BlasImplementation impl      = BlasImplementation::Basic;

    // Use our own CPU features instance
    if constexpr (std::is_same_v<T, float>) {
        if (blas_cpu_features.avx512f) {
            impl = BlasImplementation::AVX512;
        } else if (blas_cpu_features.avx2 && blas_cpu_features.fma) {
            impl = BlasImplementation::FMA;
        } else if (blas_cpu_features.avx2) {
            impl = BlasImplementation::AVX2;
        } else if (blas_cpu_features.avx) {
            impl = BlasImplementation::AVX;
        } else if (blas_cpu_features.sse4_2) {
            impl = BlasImplementation::SSE42;
        }
    }

    // Log the selected implementation
    std::string impl_str;
    switch (impl) {
    case BlasImplementation::AVX512:
        impl_str = "AVX-512";
        break;
    case BlasImplementation::FMA:
        impl_str = "FMA (AVX2+FMA)";
        break;
    case BlasImplementation::AVX2:
        impl_str = "AVX2";
        break;
    case BlasImplementation::AVX:
        impl_str = "AVX";
        break;
    case BlasImplementation::SSE42:
        impl_str = "SSE4.2";
        break;
    case BlasImplementation::Basic:
        impl_str = "Basic";
        break;
    }

    LOG_OPTIMIZATION("BLAS", impl_str, "Type: " + type_name);
    return impl;
}

//=============================================================================
// GEMM Implementation
//=============================================================================

// Main GEMM dispatcher function
template <typename T>
void BLAS<T>::gemm(bool     trans_a,
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
                   size_t   ldc) {
    // Add logging with timing for the entire GEMM operation
    TIME_OPERATION("GEMM",
                   "m=" + std::to_string(m) + ", n=" + std::to_string(n) +
                       ", k=" + std::to_string(k) + ", transA=" + std::string(trans_a ? "T" : "N") +
                       ", transB=" + std::string(trans_b ? "T" : "N"));

    // Dispatch to the appropriate implementation based on CPU features
    BlasImplementation impl = select_implementation();

    switch (impl) {
    case BlasImplementation::AVX512:
        if constexpr (std::is_same_v<T, float>) {
            gemm_avx512(trans_a, trans_b, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
            return;
        }
        break;

    case BlasImplementation::FMA:
        if constexpr (std::is_same_v<T, float>) {
            gemm_fma(trans_a, trans_b, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
            return;
        }
        break;

    case BlasImplementation::AVX2:
        if constexpr (std::is_same_v<T, float>) {
            gemm_avx2(trans_a, trans_b, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
            return;
        }
        break;

    case BlasImplementation::AVX:
        if constexpr (std::is_same_v<T, float>) {
            gemm_avx(trans_a, trans_b, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
            return;
        }
        break;

    case BlasImplementation::SSE42:
        if constexpr (std::is_same_v<T, float>) {
            gemm_sse42(trans_a, trans_b, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
            return;
        }
        break;

    case BlasImplementation::Basic:
    default:
        // Fall through to basic implementation
        break;
    }

    // Fallback to basic implementation for unsupported types or instructions
    gemm_basic(trans_a, trans_b, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

// Tensor-based GEMM
template <typename T>
void BLAS<T>::gemm(bool             trans_a,
                   bool             trans_b,
                   T                alpha,
                   const Tensor<T>& A,
                   const Tensor<T>& B,
                   T                beta,
                   Tensor<T>&       C) {
    // Log the tensor dimensions and memory layout
    LOG_DEBUG("Tensor GEMM: A shape=[",
              A.dims()[0],
              ",",
              A.dims()[1],
              "], ",
              "B shape=[",
              B.dims()[0],
              ",",
              B.dims()[1],
              "], ",
              "C shape=[",
              C.dims()[0],
              ",",
              C.dims()[1],
              "], ",
              "A layout=",
              A.layout() == MemoryLayout::RowMajor ? "RowMajor" : "ColumnMajor",
              ", ",
              "B layout=",
              B.layout() == MemoryLayout::RowMajor ? "RowMajor" : "ColumnMajor",
              ", ",
              "C layout=",
              C.layout() == MemoryLayout::RowMajor ? "RowMajor" : "ColumnMajor");

    // Extract dimensions
    size_t m, n, k, lda, ldb, ldc;

    // Handle different memory layouts
    if (A.layout() == MemoryLayout::RowMajor) {
        m   = A.dims()[0];
        k   = A.dims()[1];
        lda = trans_a ? m : k;
    } else { // ColumnMajor
        k   = A.dims()[0];
        m   = A.dims()[1];
        lda = trans_a ? k : m;
    }

    if (B.layout() == MemoryLayout::RowMajor) {
        if (trans_b) {
            n = B.dims()[0];
            assert(B.dims()[1] == k);
        } else {
            assert(B.dims()[0] == k);
            n = B.dims()[1];
        }
        ldb = trans_b ? n : k;
    } else { // ColumnMajor
        if (trans_b) {
            assert(B.dims()[0] == k);
            n = B.dims()[1];
        } else {
            n = B.dims()[0];
            assert(B.dims()[1] == k);
        }
        ldb = trans_b ? k : n;
    }

    // Check C dimensions
    if (C.layout() == MemoryLayout::RowMajor) {
        assert(C.dims()[0] == m);
        assert(C.dims()[1] == n);
        ldc = n;
    } else { // ColumnMajor
        assert(C.dims()[0] == n);
        assert(C.dims()[1] == m);
        ldc = m;
    }

    // Determine alignment for potential optimizations
    size_t alignment_A = A.alignment();
    size_t alignment_B = B.alignment();
    size_t alignment_C = C.alignment();

    // Log the dimensions we'll use for the raw pointer GEMM call
    LOG_DEBUG("Calling raw GEMM with: m=",
              m,
              ", n=",
              n,
              ", k=",
              k,
              ", lda=",
              lda,
              ", ldb=",
              ldb,
              ", ldc=",
              ldc);

    // Start timing the GEMM operation
    TIME_OPERATION("Tensor GEMM",
                   "m=" + std::to_string(m) + ", n=" + std::to_string(n) +
                       ", k=" + std::to_string(k));

    // Call the raw pointer GEMM with appropriate strides
    gemm(trans_a, trans_b, m, n, k, alpha, A.data(), lda, B.data(), ldb, beta, C.data(), ldc);
}

//=============================================================================
// GEMM Implementation Variants
//=============================================================================

// Basic GEMM implementation with OpenMP parallelization and SIMD hints
template <typename T>
void BLAS<T>::gemm_basic(bool     trans_a,
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
                         size_t   ldc) {
    // Validate dimensions
    const size_t A_rows = trans_a ? k : m;
    const size_t A_cols = trans_a ? m : k;
    const size_t B_rows = trans_b ? n : k;
    const size_t B_cols = trans_b ? k : n;

    assert(A_cols == B_rows);

    // Scale C by beta
    if (beta != static_cast<T>(1)) {
        if (beta == static_cast<T>(0)) {
#pragma omp parallel for
            for (size_t i = 0; i < m; ++i) {
#pragma omp simd
                for (size_t j = 0; j < n; ++j) {
                    C[i * ldc + j] = static_cast<T>(0);
                }
            }
        } else {
#pragma omp parallel for
            for (size_t i = 0; i < m; ++i) {
#pragma omp simd
                for (size_t j = 0; j < n; ++j) {
                    C[i * ldc + j] *= beta;
                }
            }
        }
    }

    // Perform matrix multiplication: C = alpha * A * B + beta * C
    if (!trans_a && !trans_b) {
// Standard matrix multiplication, optimized loop order for cache locality
#pragma omp parallel for
        for (size_t i = 0; i < m; ++i) {
            for (size_t p = 0; p < k; ++p) {
                const T a_val = alpha * A[i * lda + p];
#pragma omp simd
                for (size_t j = 0; j < n; ++j) {
                    C[i * ldc + j] += a_val * B[p * ldb + j];
                }
            }
        }
    } else if (trans_a && !trans_b) {
#pragma omp parallel for
        for (size_t i = 0; i < m; ++i) {
            for (size_t p = 0; p < k; ++p) {
                const T a_val = alpha * A[p * lda + i];
#pragma omp simd
                for (size_t j = 0; j < n; ++j) {
                    C[i * ldc + j] += a_val * B[p * ldb + j];
                }
            }
        }
    } else if (!trans_a && trans_b) {
#pragma omp parallel for
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                T sum = static_cast<T>(0);
#pragma omp simd reduction(+ : sum)
                for (size_t p = 0; p < k; ++p) {
                    sum += A[i * lda + p] * B[j * ldb + p];
                }
                C[i * ldc + j] += alpha * sum;
            }
        }
    } else { // trans_a && trans_b
#pragma omp parallel for
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                T sum = static_cast<T>(0);
#pragma omp simd reduction(+ : sum)
                for (size_t p = 0; p < k; ++p) {
                    sum += A[p * lda + i] * B[j * ldb + p];
                }
                C[i * ldc + j] += alpha * sum;
            }
        }
    }
}

// SSE4.2 implementation for float GEMM
template <typename T>
void BLAS<T>::gemm_sse42(bool     trans_a,
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
                         size_t   ldc) {
    // SSE4.2 implementation for float type (4 floats per operation)
    if constexpr (std::is_same_v<T, float>) {
        // For transposed matrices, use the basic implementation with SIMD hints
        if (trans_a || trans_b) {
            gemm_basic(trans_a, trans_b, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
            return;
        }

        // Scale C by beta
        if (beta != 1.0f) {
            if (beta == 0.0f) {
#pragma omp parallel for
                for (size_t i = 0; i < m; ++i) {
                    for (size_t j = 0; j < n; j += 4) {
                        if (j + 4 <= n) {
                            _mm_storeu_ps(&C[i * ldc + j], _mm_setzero_ps());
                        } else {
                            for (size_t jj = j; jj < n; ++jj) {
                                C[i * ldc + jj] = 0.0f;
                            }
                        }
                    }
                }
            } else {
                const __m128 beta_vec = _mm_set1_ps(beta);
#pragma omp parallel for
                for (size_t i = 0; i < m; ++i) {
                    for (size_t j = 0; j < n; j += 4) {
                        if (j + 4 <= n) {
                            __m128 c_vec = _mm_loadu_ps(&C[i * ldc + j]);
                            c_vec        = _mm_mul_ps(c_vec, beta_vec);
                            _mm_storeu_ps(&C[i * ldc + j], c_vec);
                        } else {
                            for (size_t jj = j; jj < n; ++jj) {
                                C[i * ldc + jj] *= beta;
                            }
                        }
                    }
                }
            }
        }

        // Perform matrix multiplication using SSE4.2
        const __m128 alpha_vec = _mm_set1_ps(alpha);

#pragma omp parallel for
        for (size_t i = 0; i < m; ++i) {
            for (size_t p = 0; p < k; ++p) {
                const __m128 a_val = _mm_set1_ps(A[i * lda + p]);
                for (size_t j = 0; j < n; j += 4) {
                    if (j + 4 <= n) {
                        __m128 b_vec = _mm_loadu_ps(&B[p * ldb + j]);
                        __m128 c_vec = _mm_loadu_ps(&C[i * ldc + j]);

                        // c += alpha * a * b
                        b_vec = _mm_mul_ps(b_vec, a_val);
                        b_vec = _mm_mul_ps(b_vec, alpha_vec);
                        c_vec = _mm_add_ps(c_vec, b_vec);

                        _mm_storeu_ps(&C[i * ldc + j], c_vec);
                    } else {
                        for (size_t jj = j; jj < n; ++jj) {
                            C[i * ldc + jj] += alpha * A[i * lda + p] * B[p * ldb + jj];
                        }
                    }
                }
            }
        }
    } else {
        // For non-float types, use the basic implementation
        gemm_basic(trans_a, trans_b, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }
}

// AVX implementation for float GEMM
template <typename T>
void BLAS<T>::gemm_avx(bool     trans_a,
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
                       size_t   ldc) {
    // AVX implementation for float type (8 floats per operation)
    if constexpr (std::is_same_v<T, float>) {
        // For transposed matrices, use the basic implementation with SIMD hints
        if (trans_a || trans_b) {
            gemm_basic(trans_a, trans_b, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
            return;
        }

        // Scale C by beta
        if (beta != 1.0f) {
            if (beta == 0.0f) {
#pragma omp parallel for
                for (size_t i = 0; i < m; ++i) {
                    for (size_t j = 0; j < n; j += 8) {
                        if (j + 8 <= n) {
                            _mm256_storeu_ps(&C[i * ldc + j], _mm256_setzero_ps());
                        } else {
                            for (size_t jj = j; jj < n; ++jj) {
                                C[i * ldc + jj] = 0.0f;
                            }
                        }
                    }
                }
            } else {
                const __m256 beta_vec = _mm256_set1_ps(beta);
#pragma omp parallel for
                for (size_t i = 0; i < m; ++i) {
                    for (size_t j = 0; j < n; j += 8) {
                        if (j + 8 <= n) {
                            __m256 c_vec = _mm256_loadu_ps(&C[i * ldc + j]);
                            c_vec        = _mm256_mul_ps(c_vec, beta_vec);
                            _mm256_storeu_ps(&C[i * ldc + j], c_vec);
                        } else {
                            for (size_t jj = j; jj < n; ++jj) {
                                C[i * ldc + jj] *= beta;
                            }
                        }
                    }
                }
            }
        }

        // Perform matrix multiplication using AVX
        const __m256 alpha_vec = _mm256_set1_ps(alpha);

#pragma omp parallel for
        for (size_t i = 0; i < m; ++i) {
            for (size_t p = 0; p < k; ++p) {
                const __m256 a_val = _mm256_set1_ps(A[i * lda + p]);
                for (size_t j = 0; j < n; j += 8) {
                    if (j + 8 <= n) {
                        __m256 b_vec = _mm256_loadu_ps(&B[p * ldb + j]);
                        __m256 c_vec = _mm256_loadu_ps(&C[i * ldc + j]);

                        // c += alpha * a * b
                        b_vec = _mm256_mul_ps(b_vec, a_val);
                        b_vec = _mm256_mul_ps(b_vec, alpha_vec);
                        c_vec = _mm256_add_ps(c_vec, b_vec);

                        _mm256_storeu_ps(&C[i * ldc + j], c_vec);
                    } else {
                        for (size_t jj = j; jj < n; ++jj) {
                            C[i * ldc + jj] += alpha * A[i * lda + p] * B[p * ldb + jj];
                        }
                    }
                }
            }
        }
    } else {
        // For non-float types, use the basic implementation
        gemm_basic(trans_a, trans_b, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }
}

// AVX2 implementation for float GEMM
template <typename T>
void BLAS<T>::gemm_avx2(bool     trans_a,
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
                        size_t   ldc) {
    // AVX2 has the same register width as AVX but includes additional
    // integer instructions and improved throughput - for this GEMM implementation
    // we'll use the same approach as AVX but with potential optimizations

    if constexpr (std::is_same_v<T, float>) {
        // For transposed matrices, use the basic implementation with SIMD hints
        if (trans_a || trans_b) {
            gemm_basic(trans_a, trans_b, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
            return;
        }

        // Scale C by beta
        if (beta != 1.0f) {
            if (beta == 0.0f) {
#pragma omp parallel for
                for (size_t i = 0; i < m; ++i) {
                    size_t j = 0;
                    // Process blocks of 32 elements at a time for better throughput
                    for (; j + 32 <= n; j += 32) {
                        _mm256_storeu_ps(&C[i * ldc + j], _mm256_setzero_ps());
                        _mm256_storeu_ps(&C[i * ldc + j + 8], _mm256_setzero_ps());
                        _mm256_storeu_ps(&C[i * ldc + j + 16], _mm256_setzero_ps());
                        _mm256_storeu_ps(&C[i * ldc + j + 24], _mm256_setzero_ps());
                    }
                    // Process remaining elements in blocks of 8
                    for (; j + 8 <= n; j += 8) {
                        _mm256_storeu_ps(&C[i * ldc + j], _mm256_setzero_ps());
                    }
                    // Process remaining elements individually
                    for (; j < n; ++j) {
                        C[i * ldc + j] = 0.0f;
                    }
                }
            } else {
                const __m256 beta_vec = _mm256_set1_ps(beta);
#pragma omp parallel for
                for (size_t i = 0; i < m; ++i) {
                    size_t j = 0;
                    // Process blocks of 32 elements at a time
                    for (; j + 32 <= n; j += 32) {
                        __m256 c_vec1 = _mm256_loadu_ps(&C[i * ldc + j]);
                        __m256 c_vec2 = _mm256_loadu_ps(&C[i * ldc + j + 8]);
                        __m256 c_vec3 = _mm256_loadu_ps(&C[i * ldc + j + 16]);
                        __m256 c_vec4 = _mm256_loadu_ps(&C[i * ldc + j + 24]);

                        c_vec1 = _mm256_mul_ps(c_vec1, beta_vec);
                        c_vec2 = _mm256_mul_ps(c_vec2, beta_vec);
                        c_vec3 = _mm256_mul_ps(c_vec3, beta_vec);
                        c_vec4 = _mm256_mul_ps(c_vec4, beta_vec);

                        _mm256_storeu_ps(&C[i * ldc + j], c_vec1);
                        _mm256_storeu_ps(&C[i * ldc + j + 8], c_vec2);
                        _mm256_storeu_ps(&C[i * ldc + j + 16], c_vec3);
                        _mm256_storeu_ps(&C[i * ldc + j + 24], c_vec4);
                    }
                    // Process remaining elements in blocks of 8
                    for (; j + 8 <= n; j += 8) {
                        __m256 c_vec = _mm256_loadu_ps(&C[i * ldc + j]);
                        c_vec        = _mm256_mul_ps(c_vec, beta_vec);
                        _mm256_storeu_ps(&C[i * ldc + j], c_vec);
                    }
                    // Process remaining elements individually
                    for (; j < n; ++j) {
                        C[i * ldc + j] *= beta;
                    }
                }
            }
        }

        // Perform matrix multiplication using AVX2 with blocking for better cache usage
        const __m256 alpha_vec = _mm256_set1_ps(alpha);

        // Block sizes for cache optimization
        const size_t block_size_m = 64; // Adjust based on cache size
        const size_t block_size_n = 256;
        const size_t block_size_k = 64;

#pragma omp parallel for
        for (size_t i_block = 0; i_block < m; i_block += block_size_m) {
            const size_t i_end = std::min(i_block + block_size_m, m);

            for (size_t j_block = 0; j_block < n; j_block += block_size_n) {
                const size_t j_end = std::min(j_block + block_size_n, n);

                for (size_t k_block = 0; k_block < k; k_block += block_size_k) {
                    const size_t k_end = std::min(k_block + block_size_k, k);

                    for (size_t i = i_block; i < i_end; ++i) {
                        for (size_t p = k_block; p < k_end; ++p) {
                            const __m256 a_val = _mm256_set1_ps(alpha * A[i * lda + p]);
                            size_t       j     = j_block;

                            // Process blocks of 32 elements at a time
                            for (; j + 32 <= j_end; j += 32) {
                                __m256 b_vec1 = _mm256_loadu_ps(&B[p * ldb + j]);
                                __m256 b_vec2 = _mm256_loadu_ps(&B[p * ldb + j + 8]);
                                __m256 b_vec3 = _mm256_loadu_ps(&B[p * ldb + j + 16]);
                                __m256 b_vec4 = _mm256_loadu_ps(&B[p * ldb + j + 24]);

                                __m256 c_vec1 = _mm256_loadu_ps(&C[i * ldc + j]);
                                __m256 c_vec2 = _mm256_loadu_ps(&C[i * ldc + j + 8]);
                                __m256 c_vec3 = _mm256_loadu_ps(&C[i * ldc + j + 16]);
                                __m256 c_vec4 = _mm256_loadu_ps(&C[i * ldc + j + 24]);

                                c_vec1 = _mm256_fmadd_ps(a_val, b_vec1, c_vec1);
                                c_vec2 = _mm256_fmadd_ps(a_val, b_vec2, c_vec2);
                                c_vec3 = _mm256_fmadd_ps(a_val, b_vec3, c_vec3);
                                c_vec4 = _mm256_fmadd_ps(a_val, b_vec4, c_vec4);

                                _mm256_storeu_ps(&C[i * ldc + j], c_vec1);
                                _mm256_storeu_ps(&C[i * ldc + j + 8], c_vec2);
                                _mm256_storeu_ps(&C[i * ldc + j + 16], c_vec3);
                                _mm256_storeu_ps(&C[i * ldc + j + 24], c_vec4);
                            }

                            // Process remaining elements in blocks of 8
                            for (; j + 8 <= j_end; j += 8) {
                                __m256 b_vec = _mm256_loadu_ps(&B[p * ldb + j]);
                                __m256 c_vec = _mm256_loadu_ps(&C[i * ldc + j]);

                                c_vec = _mm256_fmadd_ps(a_val, b_vec, c_vec);

                                _mm256_storeu_ps(&C[i * ldc + j], c_vec);
                            }

                            // Process remaining elements individually
                            for (; j < j_end; ++j) {
                                C[i * ldc + j] += alpha * A[i * lda + p] * B[p * ldb + j];
                            }
                        }
                    }
                }
            }
        }
    } else {
        // For non-float types, use the basic implementation
        gemm_basic(trans_a, trans_b, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }
}

// FMA implementation (AVX2 + FMA instructions) for float GEMM
template <typename T>
void BLAS<T>::gemm_fma(bool     trans_a,
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
                       size_t   ldc) {
    // For this implementation, we can largely reuse the AVX2 implementation
    // since we've already included FMA instructions there
    gemm_avx2(trans_a, trans_b, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

// AVX-512 implementation for float GEMM
template <typename T>
void BLAS<T>::gemm_avx512(bool     trans_a,
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
                          size_t   ldc) {
    // AVX-512 implementation for float type (16 floats per operation)
    if constexpr (std::is_same_v<T, float>) {
        // For transposed matrices, use the basic implementation with SIMD hints
        if (trans_a || trans_b) {
            gemm_basic(trans_a, trans_b, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
            return;
        }

        // Scale C by beta
        if (beta != 1.0f) {
            if (beta == 0.0f) {
#pragma omp parallel for
                for (size_t i = 0; i < m; ++i) {
                    size_t j = 0;
                    // Process blocks of 64 elements at a time
                    for (; j + 64 <= n; j += 64) {
                        _mm512_storeu_ps(&C[i * ldc + j], _mm512_setzero_ps());
                        _mm512_storeu_ps(&C[i * ldc + j + 16], _mm512_setzero_ps());
                        _mm512_storeu_ps(&C[i * ldc + j + 32], _mm512_setzero_ps());
                        _mm512_storeu_ps(&C[i * ldc + j + 48], _mm512_setzero_ps());
                    }
                    // Process remaining elements in blocks of 16
                    for (; j + 16 <= n; j += 16) {
                        _mm512_storeu_ps(&C[i * ldc + j], _mm512_setzero_ps());
                    }
                    // Process remaining elements individually
                    for (; j < n; ++j) {
                        C[i * ldc + j] = 0.0f;
                    }
                }
            } else {
                const __m512 beta_vec = _mm512_set1_ps(beta);
#pragma omp parallel for
                for (size_t i = 0; i < m; ++i) {
                    size_t j = 0;
                    // Process blocks of 64 elements at a time
                    for (; j + 64 <= n; j += 64) {
                        __m512 c_vec1 = _mm512_loadu_ps(&C[i * ldc + j]);
                        __m512 c_vec2 = _mm512_loadu_ps(&C[i * ldc + j + 16]);
                        __m512 c_vec3 = _mm512_loadu_ps(&C[i * ldc + j + 32]);
                        __m512 c_vec4 = _mm512_loadu_ps(&C[i * ldc + j + 48]);

                        c_vec1 = _mm512_mul_ps(c_vec1, beta_vec);
                        c_vec2 = _mm512_mul_ps(c_vec2, beta_vec);
                        c_vec3 = _mm512_mul_ps(c_vec3, beta_vec);
                        c_vec4 = _mm512_mul_ps(c_vec4, beta_vec);

                        _mm512_storeu_ps(&C[i * ldc + j], c_vec1);
                        _mm512_storeu_ps(&C[i * ldc + j + 16], c_vec2);
                        _mm512_storeu_ps(&C[i * ldc + j + 32], c_vec3);
                        _mm512_storeu_ps(&C[i * ldc + j + 48], c_vec4);
                    }
                    // Process remaining elements in blocks of 16
                    for (; j + 16 <= n; j += 16) {
                        __m512 c_vec = _mm512_loadu_ps(&C[i * ldc + j]);
                        c_vec        = _mm512_mul_ps(c_vec, beta_vec);
                        _mm512_storeu_ps(&C[i * ldc + j], c_vec);
                    }
                    // Process remaining elements individually
                    for (; j < n; ++j) {
                        C[i * ldc + j] *= beta;
                    }
                }
            }
        }

        // Perform matrix multiplication using AVX-512 with blocking for better cache usage
        const __m512 alpha_vec = _mm512_set1_ps(alpha);

        // Block sizes for cache optimization
        const size_t block_size_m = 64;
        const size_t block_size_n = 512;
        const size_t block_size_k = 64;

#pragma omp parallel for
        for (size_t i_block = 0; i_block < m; i_block += block_size_m) {
            const size_t i_end = std::min(i_block + block_size_m, m);

            for (size_t j_block = 0; j_block < n; j_block += block_size_n) {
                const size_t j_end = std::min(j_block + block_size_n, n);

                for (size_t k_block = 0; k_block < k; k_block += block_size_k) {
                    const size_t k_end = std::min(k_block + block_size_k, k);

                    for (size_t i = i_block; i < i_end; ++i) {
                        for (size_t p = k_block; p < k_end; ++p) {
                            const __m512 a_val = _mm512_set1_ps(alpha * A[i * lda + p]);
                            size_t       j     = j_block;

                            // Process blocks of 64 elements at a time
                            for (; j + 64 <= j_end; j += 64) {
                                __m512 b_vec1 = _mm512_loadu_ps(&B[p * ldb + j]);
                                __m512 b_vec2 = _mm512_loadu_ps(&B[p * ldb + j + 16]);
                                __m512 b_vec3 = _mm512_loadu_ps(&B[p * ldb + j + 32]);
                                __m512 b_vec4 = _mm512_loadu_ps(&B[p * ldb + j + 48]);

                                __m512 c_vec1 = _mm512_loadu_ps(&C[i * ldc + j]);
                                __m512 c_vec2 = _mm512_loadu_ps(&C[i * ldc + j + 16]);
                                __m512 c_vec3 = _mm512_loadu_ps(&C[i * ldc + j + 32]);
                                __m512 c_vec4 = _mm512_loadu_ps(&C[i * ldc + j + 48]);

                                // c += a * b (using FMA)
                                c_vec1 = _mm512_fmadd_ps(a_val, b_vec1, c_vec1);
                                c_vec2 = _mm512_fmadd_ps(a_val, b_vec2, c_vec2);
                                c_vec3 = _mm512_fmadd_ps(a_val, b_vec3, c_vec3);
                                c_vec4 = _mm512_fmadd_ps(a_val, b_vec4, c_vec4);

                                _mm512_storeu_ps(&C[i * ldc + j], c_vec1);
                                _mm512_storeu_ps(&C[i * ldc + j + 16], c_vec2);
                                _mm512_storeu_ps(&C[i * ldc + j + 32], c_vec3);
                                _mm512_storeu_ps(&C[i * ldc + j + 48], c_vec4);
                            }

                            // Process remaining elements in blocks of 16
                            for (; j + 16 <= j_end; j += 16) {
                                __m512 b_vec = _mm512_loadu_ps(&B[p * ldb + j]);
                                __m512 c_vec = _mm512_loadu_ps(&C[i * ldc + j]);

                                c_vec = _mm512_fmadd_ps(a_val, b_vec, c_vec);

                                _mm512_storeu_ps(&C[i * ldc + j], c_vec);
                            }

                            // Process remaining elements individually
                            for (; j < j_end; ++j) {
                                C[i * ldc + j] += alpha * A[i * lda + p] * B[p * ldb + j];
                            }
                        }
                    }
                }
            }
        }
    } else {
        // For non-float types, use the basic implementation
        gemm_basic(trans_a, trans_b, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }
}

//=============================================================================
// im2col Implementation
//=============================================================================

// Main im2col dispatcher function
template <typename T>
void BLAS<T>::im2col(const T* data,
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
                     T*       col_data) {
    // Log the operation dimensions
    LOG_DEBUG("im2col: batch=",
              batch_size,
              ", channels=",
              channels,
              ", height=",
              height,
              ", width=",
              width,
              ", kernel=",
              kernel_h,
              "x",
              kernel_w,
              ", output=",
              output_h,
              "x",
              output_w);

    // Time the operation
    TIME_OPERATION("im2col",
                   "batch=" + std::to_string(batch_size) +
                       ", channels=" + std::to_string(channels) +
                       ", spatial=" + std::to_string(height) + "x" + std::to_string(width));

    // For im2col, we have a basic implementation and a SIMD-optimized version
    // Determine if we should use SIMD based on the type and CPU features
    if constexpr (std::is_same_v<T, float>) {
        BlasImplementation impl = select_implementation();
        if (impl != BlasImplementation::Basic) {
            LOG_OPTIMIZATION("im2col",
                             "SIMD-optimized",
                             "batch=" + std::to_string(batch_size) +
                                 ", channels=" + std::to_string(channels));

            im2col_simd(data,
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
                        col_data);
            return;
        }
    }

    // Log using basic implementation
    LOG_OPTIMIZATION("im2col",
                     "Basic",
                     "batch=" + std::to_string(batch_size) +
                         ", channels=" + std::to_string(channels));

    // Fallback to basic implementation
    im2col_basic(data,
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
                 col_data);
}

// Tensor-based im2col
template <typename T>
void BLAS<T>::im2col(const Tensor<T>& input,
                     size_t           kernel_h,
                     size_t           kernel_w,
                     size_t           pad_h,
                     size_t           pad_w,
                     size_t           stride_h,
                     size_t           stride_w,
                     Tensor<T>&       col_output) {
    // Extract dimensions from the input tensor
    assert(input.rank() == 4); // batch_size, channels, height, width

    size_t batch_size = input.dims()[0];
    size_t channels   = input.dims()[1];
    size_t height     = input.dims()[2];
    size_t width      = input.dims()[3];

    // Calculate output dimensions
    size_t output_h = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    size_t output_w = (width + 2 * pad_w - kernel_w) / stride_w + 1;

    // Ensure the output tensor has the correct shape
    // Shape should be [batch_size, channels * kernel_h * kernel_w, output_h * output_w]
    std::vector<size_t> expected_shape = {
        batch_size, channels * kernel_h * kernel_w, output_h * output_w};

    // Verify the output tensor has the correct dimensions
    assert(col_output.rank() == 3);
    assert(col_output.dims()[0] == expected_shape[0]);
    assert(col_output.dims()[1] == expected_shape[1]);
    assert(col_output.dims()[2] == expected_shape[2]);

    // Call the raw pointer version
    im2col(input.data(),
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
           col_output.data());
}

// Basic im2col implementation
template <typename T>
void BLAS<T>::im2col_basic(const T* data,
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
                           T*       col_data) {
    // Calculate total elements in the column buffer
    const size_t channels_col = channels * kernel_h * kernel_w;
    const size_t col_size     = channels_col * output_h * output_w;

    // Initialize col_data to zeros to handle padding implicitly
    std::fill(col_data, col_data + col_size * batch_size, static_cast<T>(0));

// Process one batch at a time
#pragma omp parallel for
    for (size_t b = 0; b < batch_size; ++b) {
        // Calculate base offset for current batch
        const T* batch_data     = data + b * channels * height * width;
        T*       batch_col_data = col_data + b * channels_col * output_h * output_w;

        // Loop over each patch
        for (size_t c = 0; c < channels; ++c) {
            // Pointer to current channel data
            const T* channel_data = batch_data + c * height * width;

            for (size_t kh = 0; kh < kernel_h; ++kh) {
                for (size_t kw = 0; kw < kernel_w; ++kw) {
                    // Current position in kernel
                    const size_t col_channel = (c * kernel_h + kh) * kernel_w + kw;

                    for (size_t h = 0; h < output_h; ++h) {
                        // Input height with stride and padding offset
                        const size_t h_pad = h * stride_h - pad_h + kh;

                        // Skip if outside padded input height
                        if (h_pad >= height || h_pad < 0)
                            continue;

                        for (size_t w = 0; w < output_w; ++w) {
                            // Input width with stride and padding offset
                            const size_t w_pad = w * stride_w - pad_w + kw;

                            // Skip if outside padded input width
                            if (w_pad >= width || w_pad < 0)
                                continue;

                            const size_t col_index    = (col_channel * output_h + h) * output_w + w;
                            const size_t data_index   = h_pad * width + w_pad;
                            batch_col_data[col_index] = channel_data[data_index];
                        }
                    }
                }
            }
        }
    }
}

// SIMD-optimized im2col implementation
template <typename T>
void BLAS<T>::im2col_simd(const T* data,
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
                          T*       col_data) {
    // Calculate total elements in the column buffer
    const size_t channels_col = channels * kernel_h * kernel_w;
    const size_t col_size     = channels_col * output_h * output_w;

// Initialize col_data to zeros to handle padding implicitly
#pragma omp parallel for
    for (size_t i = 0; i < batch_size * col_size; i += 16) {
        if constexpr (std::is_same_v<T, float>) {
            if (i + 16 <= batch_size * col_size) {
                _mm512_storeu_ps(&col_data[i], _mm512_setzero_ps());
            } else {
                for (size_t j = i; j < batch_size * col_size; ++j) {
                    col_data[j] = static_cast<T>(0);
                }
            }
        } else {
            for (size_t j = i; j < std::min(i + 16, batch_size * col_size); ++j) {
                col_data[j] = static_cast<T>(0);
            }
        }
    }

    // Pre-compute indices to avoid redundant calculations
    std::vector<size_t>              kernel_indices(kernel_h * kernel_w * channels);
    std::vector<std::vector<size_t>> input_indices(output_h * output_w);

    for (size_t h = 0; h < output_h; ++h) {
        for (size_t w = 0; w < output_w; ++w) {
            std::vector<size_t>& indices = input_indices[h * output_w + w];
            indices.resize(kernel_h * kernel_w * channels, -1); // -1 indicates out of bounds

            for (size_t c = 0; c < channels; ++c) {
                for (size_t kh = 0; kh < kernel_h; ++kh) {
                    // Input height with stride and padding offset
                    const size_t h_pad = h * stride_h - pad_h + kh;

                    if (h_pad >= height || h_pad < 0) {
                        continue; // Out of bounds, keep as -1
                    }

                    for (size_t kw = 0; kw < kernel_w; ++kw) {
                        // Input width with stride and padding offset
                        const size_t w_pad = w * stride_w - pad_w + kw;

                        if (w_pad >= width || w_pad < 0) {
                            continue; // Out of bounds, keep as -1
                        }

                        const size_t kernel_idx = (c * kernel_h + kh) * kernel_w + kw;
                        indices[kernel_idx]     = c * height * width + h_pad * width + w_pad;
                    }
                }
            }
        }
    }

// Process one batch at a time
#pragma omp parallel for
    for (size_t b = 0; b < batch_size; ++b) {
        // Calculate base offset for current batch
        const T* batch_data     = data + b * channels * height * width;
        T*       batch_col_data = col_data + b * channels_col * output_h * output_w;

// Process in parallel over output positions
#pragma omp parallel for collapse(2)
        for (size_t h = 0; h < output_h; ++h) {
            for (size_t w = 0; w < output_w; ++w) {
                const size_t               output_idx = h * output_w + w;
                const std::vector<size_t>& indices    = input_indices[output_idx];

                for (size_t kernel_idx = 0; kernel_idx < kernel_h * kernel_w * channels;
                     ++kernel_idx) {
                    const size_t col_idx = kernel_idx * output_h * output_w + output_idx;

                    if (indices[kernel_idx] != static_cast<size_t>(-1)) {
                        batch_col_data[col_idx] = batch_data[indices[kernel_idx]];
                    }
                }
            }
        }
    }
}

//=============================================================================
// col2im Implementation
//=============================================================================

// Main col2im dispatcher function
template <typename T>
void BLAS<T>::col2im(const T* col_data,
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
                     T*       data) {
    // Log the operation dimensions
    LOG_DEBUG("col2im: batch=",
              batch_size,
              ", channels=",
              channels,
              ", height=",
              height,
              ", width=",
              width,
              ", kernel=",
              kernel_h,
              "x",
              kernel_w,
              ", output=",
              output_h,
              "x",
              output_w);

    // Time the operation
    TIME_OPERATION("col2im",
                   "batch=" + std::to_string(batch_size) +
                       ", channels=" + std::to_string(channels) +
                       ", spatial=" + std::to_string(height) + "x" + std::to_string(width));

    // For col2im, we have a basic implementation and a SIMD-optimized version
    // Determine if we should use SIMD based on the type and CPU features
    if constexpr (std::is_same_v<T, float>) {
        BlasImplementation impl = select_implementation();
        if (impl != BlasImplementation::Basic) {
            LOG_OPTIMIZATION("col2im",
                             "SIMD-optimized",
                             "batch=" + std::to_string(batch_size) +
                                 ", channels=" + std::to_string(channels));

            col2im_simd(col_data,
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
                        data);
            return;
        }
    }

    // Log using basic implementation
    LOG_OPTIMIZATION("col2im",
                     "Basic",
                     "batch=" + std::to_string(batch_size) +
                         ", channels=" + std::to_string(channels));

    // Fallback to basic implementation
    col2im_basic(col_data,
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
                 data);
}

// Tensor-based col2im
template <typename T>
void BLAS<T>::col2im(const Tensor<T>& col_input,
                     size_t           channels,
                     size_t           height,
                     size_t           width,
                     size_t           kernel_h,
                     size_t           kernel_w,
                     size_t           pad_h,
                     size_t           pad_w,
                     size_t           stride_h,
                     size_t           stride_w,
                     Tensor<T>&       output) {
    // Extract dimensions from the column tensor
    assert(col_input.rank() == 3); // batch_size, channels*kernel_h*kernel_w, output_h*output_w

    size_t batch_size      = col_input.dims()[0];
    size_t channels_kernel = col_input.dims()[1];
    size_t output_area     = col_input.dims()[2];

    // Calculate output dimensions
    size_t output_h = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    size_t output_w = (width + 2 * pad_w - kernel_w) / stride_w + 1;

    // Verify dimensions
    assert(channels_kernel == channels * kernel_h * kernel_w);
    assert(output_area == output_h * output_w);

    // Ensure the output tensor has the correct shape
    // Shape should be [batch_size, channels, height, width]
    assert(output.rank() == 4);
    assert(output.dims()[0] == batch_size);
    assert(output.dims()[1] == channels);
    assert(output.dims()[2] == height);
    assert(output.dims()[3] == width);

    // Call the raw pointer version
    col2im(col_input.data(),
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
           output.data());
}

// Basic col2im implementation
template <typename T>
void BLAS<T>::col2im_basic(const T* col_data,
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
                           T*       data) {
    // Zero the output data
    const size_t data_size = batch_size * channels * height * width;
    std::fill(data, data + data_size, static_cast<T>(0));

// Process one batch at a time
#pragma omp parallel for
    for (size_t b = 0; b < batch_size; ++b) {
        // Calculate base pointers for current batch
        T*       batch_data = data + b * channels * height * width;
        const T* batch_col_data =
            col_data + b * channels * kernel_h * kernel_w * output_h * output_w;

        // This part is hard to parallelize due to potential race conditions on the output
        for (size_t c = 0; c < channels; ++c) {
            // Pointer to current channel data
            T* channel_data = batch_data + c * height * width;

            for (size_t kh = 0; kh < kernel_h; ++kh) {
                for (size_t kw = 0; kw < kernel_w; ++kw) {
                    // Current position in kernel
                    const size_t col_channel = (c * kernel_h + kh) * kernel_w + kw;

                    for (size_t h = 0; h < output_h; ++h) {
                        // Input height with stride and padding offset
                        const size_t h_pad = h * stride_h - pad_h + kh;

                        // Skip if outside valid input height
                        if (h_pad >= height || h_pad < 0)
                            continue;

                        for (size_t w = 0; w < output_w; ++w) {
                            // Input width with stride and padding offset
                            const size_t w_pad = w * stride_w - pad_w + kw;

                            // Skip if outside valid input width
                            if (w_pad >= width || w_pad < 0)
                                continue;

                            const size_t col_index  = (col_channel * output_h + h) * output_w + w;
                            const size_t data_index = h_pad * width + w_pad;

// Use atomic add in a real OpenMP implementation to avoid race conditions
#pragma omp atomic
                            channel_data[data_index] += batch_col_data[col_index];
                        }
                    }
                }
            }
        }
    }
}

// SIMD-optimized col2im implementation
template <typename T>
void BLAS<T>::col2im_simd(const T* col_data,
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
                          T*       data) {
    // Zero the output data
    const size_t data_size = batch_size * channels * height * width;

#pragma omp parallel for
    for (size_t i = 0; i < data_size; i += 16) {
        if constexpr (std::is_same_v<T, float>) {
            if (i + 16 <= data_size) {
                _mm512_storeu_ps(&data[i], _mm512_setzero_ps());
            } else {
                for (size_t j = i; j < data_size; ++j) {
                    data[j] = static_cast<T>(0);
                }
            }
        } else {
            for (size_t j = i; j < std::min(i + 16, data_size); ++j) {
                data[j] = static_cast<T>(0);
            }
        }
    }

    // Pre-compute indices mapping
    std::vector<std::vector<std::pair<size_t, size_t>>> mapping(batch_size * channels * height *
                                                                width);

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            for (size_t kh = 0; kh < kernel_h; ++kh) {
                for (size_t kw = 0; kw < kernel_w; ++kw) {
                    const size_t col_channel = (c * kernel_h + kh) * kernel_w + kw;

                    for (size_t h = 0; h < output_h; ++h) {
                        const size_t h_pad = h * stride_h - pad_h + kh;

                        if (h_pad >= height || h_pad < 0)
                            continue;

                        for (size_t w = 0; w < output_w; ++w) {
                            const size_t w_pad = w * stride_w - pad_w + kw;

                            if (w_pad >= width || w_pad < 0)
                                continue;

                            const size_t data_index = b * channels * height * width +
                                                      c * height * width + h_pad * width + w_pad;

                            const size_t col_index =
                                b * channels * kernel_h * kernel_w * output_h * output_w +
                                col_channel * output_h * output_w + h * output_w + w;

                            mapping[data_index].push_back({col_index, 1}); // 1 is the multiplier
                        }
                    }
                }
            }
        }
    }

// Process the data using the pre-computed mapping
// This approach allows us to process each output element once without race conditions
#pragma omp parallel for
    for (size_t data_idx = 0; data_idx < data_size; ++data_idx) {
        T sum = 0;

        // Accumulate values from all contributing col elements
        for (const auto& [col_idx, multiplier] : mapping[data_idx]) {
            sum += col_data[col_idx] * multiplier;
        }

        data[data_idx] = sum;
    }
}

// Helper method to compute im2col indices
template <typename T>
void BLAS<T>::compute_im2col_indices(size_t               height,
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
                                     std::vector<size_t>& input_indices) {
    // Calculate indices for im2col/col2im operations
    // This can be used to optimize both implementations

    kernel_indices.resize(kernel_h * kernel_w);
    input_indices.resize(output_h * output_w * kernel_h * kernel_w, -1);

    size_t idx = 0;
    for (size_t kh = 0; kh < kernel_h; ++kh) {
        for (size_t kw = 0; kw < kernel_w; ++kw) {
            kernel_indices[idx++] = kh * kernel_w + kw;
        }
    }

    for (size_t h = 0; h < output_h; ++h) {
        for (size_t w = 0; w < output_w; ++w) {
            for (size_t kh = 0; kh < kernel_h; ++kh) {
                const size_t h_pad = h * stride_h - pad_h + kh;

                if (h_pad >= height)
                    continue;

                for (size_t kw = 0; kw < kernel_w; ++kw) {
                    const size_t w_pad = w * stride_w - pad_w + kw;

                    if (w_pad >= width)
                        continue;

                    if (h_pad < height && w_pad < width && h_pad >= 0 && w_pad >= 0) {
                        const size_t input_idx = h_pad * width + w_pad;
                        const size_t output_idx =
                            ((h * output_w + w) * kernel_h + kh) * kernel_w + kw;
                        input_indices[output_idx] = input_idx;
                    }
                }
            }
        }
    }
}

// Explicit template instantiations for commonly used types
template class BLAS<float>;
template class BLAS<double>;

} // namespace hpc::compute