/**
 * RMSNorm kernels - forward and backward passes
 * RMSNorm is a simpler variant of LayerNorm that doesn't center by mean
 */

#include "../include/kernels.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

namespace tiny_transformer {
namespace kernels {

// Warp-level reduction for sum
__device__ inline float warp_reduce_sum_rms(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

/**
 * RMSNorm forward kernel
 *
 * Computes: y = gamma * x / sqrt(mean(x^2) + eps)
 *
 * This is simpler than LayerNorm - no mean subtraction, just RMS normalization
 *
 * Input:  [batch_size, hidden_dim]
 * Output: [batch_size, hidden_dim]
 * gamma:  [hidden_dim]
 * rstd:   [batch_size] - output buffer for storing 1/RMS
 */
__global__ void rmsnorm_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    float* __restrict__ output,
    float* __restrict__ rstd,
    int batch_size,
    int hidden_dim,
    float eps
) {
    extern __shared__ float sdata[];

    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    int tid = threadIdx.x;

    const float* x = input + batch_idx * hidden_dim;
    float* y = output + batch_idx * hidden_dim;

    // Step 1: Compute mean of squares (RMS²)
    float thread_sum_sq = 0.0f;
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        thread_sum_sq += x[i] * x[i];
    }

    // Reduce sum of squares across block
    thread_sum_sq = warp_reduce_sum_rms(thread_sum_sq);
    if (tid % 32 == 0) {
        sdata[tid / 32] = thread_sum_sq;
    }
    __syncthreads();

    if (tid < 32) {
        float val = (tid < (blockDim.x + 31) / 32) ?
                    sdata[tid] : 0.0f;
        val = warp_reduce_sum_rms(val);
        if (tid == 0) {
            sdata[0] = val;
        }
    }
    __syncthreads();

    float mean_sq = sdata[0] / hidden_dim;
    float inv_rms = rsqrtf(mean_sq + eps);  // 1/sqrt(mean_sq + eps)

    if (tid == 0 && rstd != nullptr) {
        rstd[batch_idx] = inv_rms;
    }
    __syncthreads();

    // Step 2: Normalize and scale
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        y[i] = gamma[i] * x[i] * inv_rms;
    }
}

/**
 * RMSNorm backward kernel
 *
 * Backprop through: y = gamma * x / sqrt(mean(x^2) + eps)
 *
 * Derivation:
 * Let RMS = sqrt(mean(x^2) + eps)
 * y_i = gamma_i * x_i / RMS
 *
 * ∂L/∂x_i = gamma_i * ∂L/∂y_i / RMS
 *         - (x_i / RMS) * Σ_j (gamma_j * x_j * ∂L/∂y_j) / (N * RMS^2)
 *
 * grad_output: [batch_size, hidden_dim]
 * input:       [batch_size, hidden_dim]
 * gamma:       [hidden_dim]
 * rstd:        [batch_size] - from forward pass (1/RMS)
 * grad_input:  [batch_size, hidden_dim] - output
 * grad_gamma:  [hidden_dim] - output (accumulated)
 */
__global__ void rmsnorm_backward_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ rstd,
    float* __restrict__ grad_input,
    float* __restrict__ grad_gamma,
    int batch_size,
    int hidden_dim
) {
    extern __shared__ float sdata[];

    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    int tid = threadIdx.x;

    const float* x = input + batch_idx * hidden_dim;
    const float* dy = grad_output + batch_idx * hidden_dim;
    float* dx = grad_input + batch_idx * hidden_dim;

    float inv_rms = rstd[batch_idx];

    // Compute sum: Σ (gamma[i] * x[i] * dy[i])
    float thread_sum = 0.0f;

    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float x_norm = x[i] * inv_rms;
        thread_sum += gamma[i] * x[i] * dy[i];

        // Accumulate gradient for gamma
        if (grad_gamma != nullptr) {
            atomicAdd(&grad_gamma[i], dy[i] * x_norm);
        }
    }

    // Reduce sum
    thread_sum = warp_reduce_sum_rms(thread_sum);
    if (tid % 32 == 0) {
        sdata[tid / 32] = thread_sum;
    }
    __syncthreads();

    if (tid < 32) {
        float val = (tid < (blockDim.x + 31) / 32) ?
                    sdata[tid] : 0.0f;
        val = warp_reduce_sum_rms(val);
        if (tid == 0) {
            sdata[0] = val;
        }
    }
    __syncthreads();
    float sum_gamma_x_dy = sdata[0];

    // Compute gradient w.r.t. input
    float inv_rms_cubed = inv_rms * inv_rms * inv_rms;
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float term1 = gamma[i] * dy[i] * inv_rms;
        float term2 = x[i] * sum_gamma_x_dy * inv_rms_cubed / hidden_dim;
        dx[i] = term1 - term2;
    }
}

// Host functions

void rmsnorm_forward(
    const float* input,
    const float* gamma,
    float* output,
    float* rstd,
    int batch_size,
    int hidden_dim,
    float eps
) {
    int blockSize = 256;
    int gridSize = batch_size;
    size_t sharedMemSize = (blockSize / 32 + 1) * sizeof(float);

    rmsnorm_forward_kernel<<<gridSize, blockSize, sharedMemSize>>>(
        input, gamma, output, rstd,
        batch_size, hidden_dim, eps
    );
    CUDA_CHECK(cudaGetLastError());
}

void rmsnorm_backward(
    const float* grad_output,
    const float* input,
    const float* gamma,
    const float* rstd,
    float* grad_input,
    float* grad_gamma,
    int batch_size,
    int hidden_dim
) {
    int blockSize = 256;
    int gridSize = batch_size;
    size_t sharedMemSize = (blockSize / 32 + 1) * sizeof(float);

    // Zero out gamma gradient
    if (grad_gamma != nullptr) {
        CUDA_CHECK(cudaMemset(grad_gamma, 0, hidden_dim * sizeof(float)));
    }

    rmsnorm_backward_kernel<<<gridSize, blockSize, sharedMemSize>>>(
        grad_output, input, gamma, rstd,
        grad_input, grad_gamma,
        batch_size, hidden_dim
    );
    CUDA_CHECK(cudaGetLastError());
}

} // namespace kernels
} // namespace tiny_transformer
