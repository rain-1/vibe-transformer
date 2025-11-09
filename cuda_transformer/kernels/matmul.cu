/**
 * Matrix multiplication kernels
 * Provides both custom tiled implementation and cuBLAS wrappers
 * Adapted from MLP-cuda reference implementation
 */

#include "../include/kernels.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>

namespace tiny_transformer {
namespace kernels {

// cuBLAS handle (should be initialized once)
static cublasHandle_t cublas_handle = nullptr;

#define TILE_SIZE 16

// ============================================================================
// Custom Tiled Matrix Multiplication Kernels
// ============================================================================

/**
 * Tiled matrix multiplication: C = A * B
 * A: [M x K], B: [K x N], C: [M x N]
 */
__global__ void matmul_tiled_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N
) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tile of A into shared memory
        if (row < M && t * TILE_SIZE + threadIdx.x < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        // Load tile of B into shared memory
        if (col < N && t * TILE_SIZE + threadIdx.y < K)
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();
    }

    // Write result
    if (row < M && col < N)
        C[row * N + col] = sum;
}

/**
 * Tiled matrix multiplication with B transposed: C = A * B^T
 * A: [M x K], B: [N x K], C: [M x N]
 */
__global__ void matmul_transB_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N
) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tile of A into shared memory
        if (row < M && t * TILE_SIZE + threadIdx.x < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        // Load tile of B^T into shared memory (reading from B with swapped indices)
        if (col < N && t * TILE_SIZE + threadIdx.y < K)
            Bs[threadIdx.y][threadIdx.x] = B[col * K + t * TILE_SIZE + threadIdx.y];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();
    }

    // Write result
    if (row < M && col < N)
        C[row * N + col] = sum;
}

/**
 * Tiled matrix multiplication with A transposed: C = A^T * B
 * A: [K x M], B: [K x N], C: [M x N]
 */
__global__ void matmul_transA_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N
) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tile of A^T into shared memory (reading from A with swapped indices)
        // A is [K x M], we want A^T which is [M x K]
        if (row < M && t * TILE_SIZE + threadIdx.x < K)
            As[threadIdx.y][threadIdx.x] = A[(t * TILE_SIZE + threadIdx.x) * M + row];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        // Load tile of B into shared memory
        if (col < N && t * TILE_SIZE + threadIdx.y < K)
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();
    }

    // Write result
    if (row < M && col < N)
        C[row * N + col] = sum;
}

// ============================================================================
// Host Functions - Custom Implementations
// ============================================================================

void matmul_custom(
    const float* A, const float* B, float* C,
    int M, int K, int N
) {
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim(
        (N + TILE_SIZE - 1) / TILE_SIZE,
        (M + TILE_SIZE - 1) / TILE_SIZE
    );

    matmul_tiled_kernel<<<gridDim, blockDim>>>(A, B, C, M, K, N);
    CUDA_CHECK(cudaGetLastError());
}

void matmul_transB_custom(
    const float* A, const float* B, float* C,
    int M, int K, int N
) {
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim(
        (N + TILE_SIZE - 1) / TILE_SIZE,
        (M + TILE_SIZE - 1) / TILE_SIZE
    );

    matmul_transB_kernel<<<gridDim, blockDim>>>(A, B, C, M, K, N);
    CUDA_CHECK(cudaGetLastError());
}

void matmul_transA_custom(
    const float* A, const float* B, float* C,
    int M, int K, int N
) {
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim(
        (N + TILE_SIZE - 1) / TILE_SIZE,
        (M + TILE_SIZE - 1) / TILE_SIZE
    );

    matmul_transA_kernel<<<gridDim, blockDim>>>(A, B, C, M, K, N);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Host Functions - cuBLAS Wrappers (Faster)
// ============================================================================

void init_cublas() {
    if (cublas_handle == nullptr) {
        cublasStatus_t status = cublasCreate(&cublas_handle);
        if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "cuBLAS initialization failed\n");
        }
    }
}

void cleanup_cublas() {
    if (cublas_handle != nullptr) {
        cublasDestroy(cublas_handle);
        cublas_handle = nullptr;
    }
}

/**
 * Matrix multiplication using cuBLAS: C = alpha * A @ B + beta * C
 * A: [M x K], B: [K x N], C: [M x N]
 */
void matmul_forward(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    float alpha, float beta
) {
    init_cublas();

    // cuBLAS uses column-major, we use row-major
    // Row-major A @ B = Col-major (B^T @ A^T)^T
    // We compute: C^T = B^T @ A^T by doing gemm(B^T, A^T) in column-major
    // This is equivalent to gemm(B, A, ldb=N, lda=K, ldc=N) treating as col-major

    cublasStatus_t status = cublasSgemm(
        cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, N,  // B in column-major (transposed in row-major)
        A, K,  // A in column-major (transposed in row-major)
        &beta,
        C, N   // C in column-major (transposed in row-major)
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS matmul failed with status %d\n", status);
    }
}

/**
 * Matrix multiplication backward pass
 * Given: C = A @ B and dL/dC
 * Compute: dL/dA and dL/dB
 */
void matmul_backward(
    const float* A, const float* B, const float* dC,
    float* dA, float* dB,
    int M, int N, int K
) {
    init_cublas();
    float alpha = 1.0f, beta = 0.0f;

    if (dA != nullptr) {
        // dA = dC @ B^T  (shape: [M x N] @ [N x K] = [M x K])
        cublasSgemm(
            cublas_handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            K, M, N,
            &alpha,
            B, N,
            dC, N,
            &beta,
            dA, K
        );
    }

    if (dB != nullptr) {
        // dB = A^T @ dC  (shape: [K x M] @ [M x N] = [K x N])
        cublasSgemm(
            cublas_handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            N, K, M,
            &alpha,
            dC, N,
            A, K,
            &beta,
            dB, N
        );
    }
}

// ============================================================================
// Sum Operations
// ============================================================================

/**
 * Sum across rows using reduction in shared memory
 * Input:  [batch_size, hidden_dim]
 * Output: [hidden_dim]
 */
__global__ void sum_rows_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int hidden_dim
) {
    extern __shared__ float sdata[];

    int col = blockIdx.x;
    if (col >= hidden_dim) return;

    int tid = threadIdx.x;

    // Each thread sums a subset of rows
    float sum = 0.0f;
    for (int row = tid; row < batch_size; row += blockDim.x) {
        sum += input[row * hidden_dim + col];
    }

    sdata[tid] = sum;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[col] = sdata[0];
    }
}

void sum_rows(const float* input, float* output, int batch_size, int hidden_dim) {
    int blockSize = 256;
    int gridSize = hidden_dim;
    size_t sharedMemSize = blockSize * sizeof(float);

    sum_rows_kernel<<<gridSize, blockSize, sharedMemSize>>>(
        input, output, batch_size, hidden_dim
    );
    CUDA_CHECK(cudaGetLastError());
}

} // namespace kernels
} // namespace tiny_transformer
