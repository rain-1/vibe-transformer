/**
 * LayerNorm kernels - forward and backward passes
 * Adapted from MLP-cuda reference implementation
 */

#include "../include/kernels.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

namespace tiny_transformer {
namespace kernels {

// Warp-level reduction for sum
__device__ inline float warp_reduce_sum(float val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

/**
 * LayerNorm forward kernel
 *
 * Computes: y = gamma * (x - mean) / sqrt(variance + eps) + beta
 *
 * Input:  [batch_size, hidden_dim]
 * Output: [batch_size, hidden_dim]
 * gamma:  [hidden_dim]
 * beta:   [hidden_dim]
 * mean:   [batch_size] - output buffer for storing mean
 * rstd:   [batch_size] - output buffer for storing 1/sqrt(variance + eps)
 */
__global__ void layernorm_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    float* __restrict__ mean,
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

    // Step 1: Compute mean
    float thread_sum = 0.0f;
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        thread_sum += x[i];
    }

    // Reduce sum across block
    thread_sum = warp_reduce_sum(thread_sum);
    if (tid % warpSize == 0) {
        sdata[tid / warpSize] = thread_sum;
    }
    __syncthreads();

    if (tid < warpSize) {
        float val = (tid < (blockDim.x + warpSize - 1) / warpSize) ?
                    sdata[tid] : 0.0f;
        val = warp_reduce_sum(val);
        if (tid == 0) {
            sdata[0] = val;
        }
    }
    __syncthreads();

    float row_mean = sdata[0] / hidden_dim;
    if (tid == 0 && mean != nullptr) {
        mean[batch_idx] = row_mean;
    }
    __syncthreads();

    // Step 2: Compute variance
    float thread_var = 0.0f;
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float diff = x[i] - row_mean;
        thread_var += diff * diff;
    }

    // Reduce variance across block
    thread_var = warp_reduce_sum(thread_var);
    if (tid % warpSize == 0) {
        sdata[tid / warpSize] = thread_var;
    }
    __syncthreads();

    if (tid < warpSize) {
        float val = (tid < (blockDim.x + warpSize - 1) / warpSize) ?
                    sdata[tid] : 0.0f;
        val = warp_reduce_sum(val);
        if (tid == 0) {
            sdata[0] = val;
        }
    }
    __syncthreads();

    float variance = sdata[0] / hidden_dim;
    float inv_std = rsqrtf(variance + eps);  // 1/sqrt(variance + eps)

    if (tid == 0 && rstd != nullptr) {
        rstd[batch_idx] = inv_std;
    }
    __syncthreads();

    // Step 3: Normalize, scale, and shift
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float normalized = (x[i] - row_mean) * inv_std;
        y[i] = gamma[i] * normalized + beta[i];
    }
}

/**
 * LayerNorm backward kernel
 *
 * Backprop through: y = gamma * (x - mean) / sqrt(variance + eps) + beta
 *
 * grad_output: [batch_size, hidden_dim]
 * input:       [batch_size, hidden_dim]
 * gamma:       [hidden_dim]
 * mean:        [batch_size] - from forward pass
 * rstd:        [batch_size] - from forward pass
 * grad_input:  [batch_size, hidden_dim] - output
 * grad_gamma:  [hidden_dim] - output (accumulated)
 * grad_beta:   [hidden_dim] - output (accumulated)
 */
__global__ void layernorm_backward_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ mean,
    const float* __restrict__ rstd,
    float* __restrict__ grad_input,
    float* __restrict__ grad_gamma,
    float* __restrict__ grad_beta,
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

    float row_mean = mean[batch_idx];
    float inv_std = rstd[batch_idx];

    // Compute intermediate sums needed for gradient computation
    // sum1 = Σ dy[i] * gamma[i]
    // sum2 = Σ dy[i] * gamma[i] * x_hat[i]
    float thread_sum1 = 0.0f;
    float thread_sum2 = 0.0f;

    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float x_hat = (x[i] - row_mean) * inv_std;
        thread_sum1 += dy[i] * gamma[i];
        thread_sum2 += dy[i] * gamma[i] * x_hat;

        // Accumulate gradients for gamma and beta
        if (grad_gamma != nullptr) {
            atomicAdd(&grad_gamma[i], dy[i] * x_hat);
        }
        if (grad_beta != nullptr) {
            atomicAdd(&grad_beta[i], dy[i]);
        }
    }

    // Reduce sum1
    thread_sum1 = warp_reduce_sum(thread_sum1);
    if (tid % warpSize == 0) {
        sdata[tid / warpSize] = thread_sum1;
    }
    __syncthreads();

    if (tid < warpSize) {
        float val = (tid < (blockDim.x + warpSize - 1) / warpSize) ?
                    sdata[tid] : 0.0f;
        val = warp_reduce_sum(val);
        if (tid == 0) {
            sdata[0] = val;
        }
    }
    __syncthreads();
    float sum1 = sdata[0];

    // Reduce sum2 (use second half of shared memory)
    thread_sum2 = warp_reduce_sum(thread_sum2);
    if (tid % warpSize == 0) {
        sdata[blockDim.x / warpSize + tid / warpSize] = thread_sum2;
    }
    __syncthreads();

    if (tid < warpSize) {
        int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        float val = (tid < num_warps) ?
                    sdata[blockDim.x / warpSize + tid] : 0.0f;
        val = warp_reduce_sum(val);
        if (tid == 0) {
            sdata[1] = val;
        }
    }
    __syncthreads();
    float sum2 = sdata[1];

    // Compute gradient w.r.t. input
    // dx[i] = gamma[i] * inv_std * (dy[i] - sum1/N - x_hat[i] * sum2/N)
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float x_hat = (x[i] - row_mean) * inv_std;
        float dx_val = gamma[i] * dy[i];
        dx_val -= sum1 / hidden_dim;
        dx_val -= x_hat * sum2 / hidden_dim;
        dx_val *= inv_std;
        dx[i] = dx_val;
    }
}

// Host functions

void layernorm_forward(
    const float* input,
    const float* gamma,
    const float* beta,
    float* output,
    float* mean,
    float* rstd,
    int batch_size,
    int hidden_dim,
    float eps
) {
    int blockSize = 256;
    int gridSize = batch_size;
    size_t sharedMemSize = (blockSize / warpSize + 2) * sizeof(float);

    layernorm_forward_kernel<<<gridSize, blockSize, sharedMemSize>>>(
        input, gamma, beta, output, mean, rstd,
        batch_size, hidden_dim, eps
    );
    CUDA_CHECK(cudaGetLastError());
}

void layernorm_backward(
    const float* grad_output,
    const float* input,
    const float* gamma,
    const float* mean,
    const float* rstd,
    float* grad_input,
    float* grad_gamma,
    float* grad_beta,
    int batch_size,
    int hidden_dim
) {
    int blockSize = 256;
    int gridSize = batch_size;
    size_t sharedMemSize = (blockSize / warpSize * 2 + 2) * sizeof(float);

    // Zero out gamma and beta gradients
    if (grad_gamma != nullptr) {
        CUDA_CHECK(cudaMemset(grad_gamma, 0, hidden_dim * sizeof(float)));
    }
    if (grad_beta != nullptr) {
        CUDA_CHECK(cudaMemset(grad_beta, 0, hidden_dim * sizeof(float)));
    }

    layernorm_backward_kernel<<<gridSize, blockSize, sharedMemSize>>>(
        grad_output, input, gamma, mean, rstd,
        grad_input, grad_gamma, grad_beta,
        batch_size, hidden_dim
    );
    CUDA_CHECK(cudaGetLastError());
}

} // namespace kernels
} // namespace tiny_transformer
