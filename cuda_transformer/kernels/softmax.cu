/**
 * Numerically stable softmax kernel with backward pass.
 * Uses log-sum-exp trick for numerical stability.
 */

#include "../include/kernels.h"
#include <cuda_runtime.h>
#include <cmath>

namespace tiny_transformer {
namespace kernels {

// ============================================================================
// Forward Pass
// ============================================================================

// Warp-level reduction for max
__device__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Warp-level reduction for sum
__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Softmax forward kernel: y_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
// Each block processes one sequence
__global__ void softmax_forward_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int seq_len,
    int dim
) {
    extern __shared__ float shared[];

    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;

    const float* in_ptr = input + batch_idx * dim;
    float* out_ptr = output + batch_idx * dim;

    // Step 1: Find max value for numerical stability
    float thread_max = -INFINITY;
    for (int i = tid; i < dim; i += blockDim.x) {
        thread_max = fmaxf(thread_max, in_ptr[i]);
    }

    // Reduce max across block
    thread_max = warp_reduce_max(thread_max);
    if (tid % 32 == 0) {
        shared[tid / 32] = thread_max;
    }
    __syncthreads();

    if (tid < 32) {
        float val = (tid < (blockDim.x + 31) / 32) ?
                    shared[tid] : -INFINITY;
        val = warp_reduce_max(val);
        if (tid == 0) {
            shared[0] = val;
        }
    }
    __syncthreads();
    float max_val = shared[0];

    // Step 2: Compute exp(x - max) and sum
    float thread_sum = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        float exp_val = expf(in_ptr[i] - max_val);
        out_ptr[i] = exp_val;
        thread_sum += exp_val;
    }

    // Reduce sum across block
    thread_sum = warp_reduce_sum(thread_sum);
    if (tid % 32 == 0) {
        shared[tid / 32] = thread_sum;
    }
    __syncthreads();

    if (tid < 32) {
        float val = (tid < (blockDim.x + 31) / 32) ?
                    shared[tid] : 0.0f;
        val = warp_reduce_sum(val);
        if (tid == 0) {
            shared[0] = val;
        }
    }
    __syncthreads();
    float sum_exp = shared[0];

    // Step 3: Normalize
    for (int i = tid; i < dim; i += blockDim.x) {
        out_ptr[i] /= sum_exp;
    }
}

void softmax_forward(
    const float* input, float* output,
    int batch_size, int seq_len, int dim
) {
    int total_sequences = batch_size * seq_len;
    int threads = 256;
    int shared_mem = (threads / 32) * sizeof(float);

    softmax_forward_kernel<<<total_sequences, threads, shared_mem>>>(
        input, output, seq_len, dim
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Softmax forward error: %s\n", cudaGetErrorString(err));
    }
}

// ============================================================================
// Backward Pass
// ============================================================================

// Softmax backward:
// dy/dx_i = y_i * (dL/dy_i - sum_j(y_j * dL/dy_j))
//
// Derivation:
// For softmax y_i = exp(x_i) / sum_j(exp(x_j)), the Jacobian is:
// dy_i/dx_j = y_i * (delta_ij - y_j)
// where delta_ij is Kronecker delta
//
// Therefore:
// dL/dx_i = sum_j (dL/dy_j * dy_j/dx_i)
//         = sum_j (dL/dy_j * y_j * (delta_ij - y_i))
//         = y_i * (dL/dy_i - sum_j(y_j * dL/dy_j))

__global__ void softmax_backward_kernel(
    const float* __restrict__ output,      // softmax output (y)
    const float* __restrict__ grad_output, // gradient wrt output (dL/dy)
    float* __restrict__ grad_input,        // gradient wrt input (dL/dx)
    int seq_len,
    int dim
) {
    extern __shared__ float shared[];

    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;

    const float* y_ptr = output + batch_idx * dim;
    const float* dy_ptr = grad_output + batch_idx * dim;
    float* dx_ptr = grad_input + batch_idx * dim;

    // Step 1: Compute sum_j(y_j * dL/dy_j)
    float thread_sum = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        thread_sum += y_ptr[i] * dy_ptr[i];
    }

    // Reduce sum across block
    thread_sum = warp_reduce_sum(thread_sum);
    if (tid % 32 == 0) {
        shared[tid / 32] = thread_sum;
    }
    __syncthreads();

    if (tid < 32) {
        float val = (tid < (blockDim.x + 31) / 32) ?
                    shared[tid] : 0.0f;
        val = warp_reduce_sum(val);
        if (tid == 0) {
            shared[0] = val;
        }
    }
    __syncthreads();
    float sum_y_dy = shared[0];

    // Step 2: Compute dL/dx_i = y_i * (dL/dy_i - sum_y_dy)
    for (int i = tid; i < dim; i += blockDim.x) {
        dx_ptr[i] = y_ptr[i] * (dy_ptr[i] - sum_y_dy);
    }
}

void softmax_backward(
    const float* output, const float* grad_output, float* grad_input,
    int batch_size, int seq_len, int dim
) {
    int total_sequences = batch_size * seq_len;
    int threads = 256;
    int shared_mem = (threads / 32) * sizeof(float);

    softmax_backward_kernel<<<total_sequences, threads, shared_mem>>>(
        output, grad_output, grad_input, seq_len, dim
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Softmax backward error: %s\n", cudaGetErrorString(err));
    }
}

} // namespace kernels
} // namespace tiny_transformer
