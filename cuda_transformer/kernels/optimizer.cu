/**
 * Optimizer kernels - AdamW implementation
 * Adapted from MLP-cuda reference implementation
 */

#include "../include/kernels.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

namespace tiny_transformer {
namespace kernels {

/**
 * AdamW optimizer update kernel
 *
 * Implements AdamW (Adam with decoupled weight decay):
 * m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
 * v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
 * m̂_t = m_t / (1 - β₁^t)
 * v̂_t = v_t / (1 - β₂^t)
 * θ_t = θ_{t-1} - lr * (m̂_t / (√v̂_t + ε) + λ * θ_{t-1})
 *
 * where the last term (λ * θ_{t-1}) is the weight decay term
 *
 * params:     [size] - model parameters
 * grads:      [size] - gradients
 * m:          [size] - first moment estimates
 * v:          [size] - second moment estimates
 * lr:         learning rate
 * beta1:      exponential decay rate for first moment (typically 0.9)
 * beta2:      exponential decay rate for second moment (typically 0.999)
 * epsilon:    small constant for numerical stability (typically 1e-8)
 * weight_decay: L2 penalty coefficient (typically 0.01)
 * beta1_corr: 1 - beta1^t (bias correction for first moment)
 * beta2_corr: 1 - beta2^t (bias correction for second moment)
 */
__global__ void adamw_update_kernel(
    float* __restrict__ params,
    const float* __restrict__ grads,
    float* __restrict__ m,
    float* __restrict__ v,
    float lr,
    float beta1,
    float beta2,
    float epsilon,
    float weight_decay,
    float beta1_corr,
    float beta2_corr,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        float grad = grads[idx];
        float param = params[idx];

        // Update biased first moment estimate
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * grad;

        // Update biased second moment estimate
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * grad * grad;

        // Compute bias-corrected moment estimates
        float m_hat = m[idx] / beta1_corr;
        float v_hat = v[idx] / beta2_corr;

        // AdamW update: separate weight decay from gradient-based update
        float update = m_hat / (sqrtf(v_hat) + epsilon);

        // Apply weight decay directly to parameters (decoupled weight decay)
        // and gradient update
        params[idx] = param - lr * (update + weight_decay * param);
    }
}

/**
 * Gradient clipping by global norm
 *
 * Computes global norm of all gradients and scales them if norm exceeds max_norm
 *
 * grad_norm² = Σ g_i²
 * if grad_norm > max_norm:
 *     g_i *= max_norm / grad_norm
 */
__global__ void compute_grad_norm_squared_kernel(
    const float* __restrict__ grads,
    float* __restrict__ partial_sums,
    int size
) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Compute local sum of squares
    float local_sum = 0.0f;
    if (idx < size) {
        float g = grads[idx];
        local_sum = g * g;
    }

    sdata[tid] = local_sum;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block
    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

__global__ void clip_grads_kernel(
    float* __restrict__ grads,
    float scale,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grads[idx] *= scale;
    }
}

// ============================================================================
// Host Functions
// ============================================================================

void adamw_update(
    float* params,
    const float* grads,
    float* m,
    float* v,
    float lr,
    float beta1,
    float beta2,
    float epsilon,
    float weight_decay,
    int t,
    int size
) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    // Compute bias correction terms: 1 - beta^t
    float beta1_t = powf(beta1, (float)t);
    float beta2_t = powf(beta2, (float)t);
    float beta1_corr = 1.0f - beta1_t;
    float beta2_corr = 1.0f - beta2_t;

    adamw_update_kernel<<<gridSize, blockSize>>>(
        params, grads, m, v,
        lr, beta1, beta2, epsilon, weight_decay,
        beta1_corr, beta2_corr,
        size
    );
    CUDA_CHECK(cudaGetLastError());
}

float compute_grad_norm(const float* grads, int size) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    size_t sharedMemSize = blockSize * sizeof(float);

    // Allocate device memory for partial sums
    float* d_partial_sums;
    CUDA_CHECK(cudaMalloc(&d_partial_sums, gridSize * sizeof(float)));

    // Compute partial sums
    compute_grad_norm_squared_kernel<<<gridSize, blockSize, sharedMemSize>>>(
        grads, d_partial_sums, size
    );
    CUDA_CHECK(cudaGetLastError());

    // Copy partial sums to host and reduce
    float* h_partial_sums = new float[gridSize];
    CUDA_CHECK(cudaMemcpy(h_partial_sums, d_partial_sums, gridSize * sizeof(float),
                         cudaMemcpyDeviceToHost));

    float total_norm_sq = 0.0f;
    for (int i = 0; i < gridSize; i++) {
        total_norm_sq += h_partial_sums[i];
    }

    // Cleanup
    delete[] h_partial_sums;
    CUDA_CHECK(cudaFree(d_partial_sums));

    return sqrtf(total_norm_sq);
}

void clip_grad_norm(float* grads, int size, float max_norm) {
    // Compute current gradient norm
    float grad_norm = compute_grad_norm(grads, size);

    // Only clip if norm exceeds max_norm
    if (grad_norm > max_norm) {
        float scale = max_norm / grad_norm;

        int blockSize = 256;
        int gridSize = (size + blockSize - 1) / blockSize;

        clip_grads_kernel<<<gridSize, blockSize>>>(grads, scale, size);
        CUDA_CHECK(cudaGetLastError());
    }
}

/**
 * Zero out gradients
 */
void zero_grads(float* grads, int size) {
    CUDA_CHECK(cudaMemset(grads, 0, size * sizeof(float)));
}

} // namespace kernels
} // namespace tiny_transformer
