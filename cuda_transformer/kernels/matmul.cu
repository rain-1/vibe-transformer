/**
 * Matrix multiplication kernels - TO BE IMPLEMENTED
 * For now, uses cuBLAS wrappers
 */

#include "../include/kernels.h"
#include <cublas_v2.h>
#include <stdio.h>

namespace tiny_transformer {
namespace kernels {

// cuBLAS handle (should be initialized once)
static cublasHandle_t cublas_handle = nullptr;

void init_cublas() {
    if (cublas_handle == nullptr) {
        cublasCreate(&cublas_handle);
    }
}

void matmul_forward(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    float alpha, float beta
) {
    init_cublas();
    
    // C = alpha * A @ B + beta * C
    // cuBLAS uses column-major, so we compute B^T @ A^T = (A @ B)^T
    cublasSgemm(cublas_handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,
                &alpha,
                B, N,
                A, K,
                &beta,
                C, N);
}

void matmul_backward(
    const float* A, const float* B, const float* dC,
    float* dA, float* dB,
    int M, int N, int K
) {
    init_cublas();
    
    // dA = dC @ B^T
    float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(cublas_handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                K, M, N,
                &alpha,
                B, N,
                dC, N,
                &beta,
                dA, K);
    
    // dB = A^T @ dC
    cublasSgemm(cublas_handle,
                CUBLAS_OP_N, CUBLAS_OP_T,
                N, K, M,
                &alpha,
                dC, N,
                A, K,
                &beta,
                dB, N);
}

} // namespace kernels
} // namespace tiny_transformer
