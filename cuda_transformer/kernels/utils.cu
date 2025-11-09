/**
 * Utility kernels - activation functions and element-wise operations
 * Adapted from MLP-cuda reference implementation
 */

#include "../include/kernels.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

namespace tiny_transformer {
namespace kernels {

// ============================================================================
// Activation Functions
// ============================================================================

/**
 * ReLU activation forward
 * y = max(0, x)
 */
__global__ void relu_forward_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

/**
 * ReLU activation backward
 * grad_input = grad_output * (input > 0)
 */
__global__ void relu_backward_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ input,
    float* __restrict__ grad_input,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_input[idx] = (input[idx] > 0.0f) ? grad_output[idx] : 0.0f;
    }
}

/**
 * GELU activation forward
 * GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
 */
__global__ void gelu_forward_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        const float sqrt_2_over_pi = 0.7978845608f;  // sqrt(2/π)
        float x_cubed = x * x * x;
        float inner = sqrt_2_over_pi * (x + 0.044715f * x_cubed);
        output[idx] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

/**
 * GELU activation backward
 */
__global__ void gelu_backward_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ input,
    float* __restrict__ grad_input,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        const float sqrt_2_over_pi = 0.7978845608f;
        float x_cubed = x * x * x;
        float inner = sqrt_2_over_pi * (x + 0.044715f * x_cubed);
        float tanh_inner = tanhf(inner);

        // Derivative: sech²(inner) = 1 - tanh²(inner)
        float sech_sq = 1.0f - tanh_inner * tanh_inner;
        float d_inner = sqrt_2_over_pi * (1.0f + 3.0f * 0.044715f * x * x);
        float d_gelu = 0.5f * (1.0f + tanh_inner) + 0.5f * x * sech_sq * d_inner;

        grad_input[idx] = grad_output[idx] * d_gelu;
    }
}

// ============================================================================
// Element-wise Operations
// ============================================================================

/**
 * Element-wise addition: C = A + B
 */
__global__ void elementwise_add_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = A[idx] + B[idx];
    }
}

/**
 * Element-wise multiplication: C = A * B
 */
__global__ void elementwise_multiply_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = A[idx] * B[idx];
    }
}

/**
 * Scale tensor by constant: A = A * scale
 */
__global__ void scale_kernel(
    float* __restrict__ A,
    float scale,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        A[idx] *= scale;
    }
}

/**
 * Add bias to each row of matrix
 * Input:  [batch_size, hidden_dim]
 * Bias:   [hidden_dim]
 * Output: [batch_size, hidden_dim]
 */
__global__ void add_bias_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int hidden_dim
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < batch_size && col < hidden_dim) {
        output[row * hidden_dim + col] = input[row * hidden_dim + col] + bias[col];
    }
}

/**
 * Residual connection with scaling: output = input + scale * residual
 */
__global__ void residual_add_kernel(
    const float* __restrict__ input,
    const float* __restrict__ residual,
    float* __restrict__ output,
    float scale,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] + scale * residual[idx];
    }
}

/**
 * Check for NaN or Inf values
 * Returns count of non-finite values
 */
__global__ void check_finite_kernel(
    const float* __restrict__ input,
    int* __restrict__ count,
    int size
) {
    extern __shared__ int sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Check if value is finite
    int local_count = 0;
    if (idx < size) {
        if (!isfinite(input[idx])) {
            local_count = 1;
        }
    }

    sdata[tid] = local_count;
    __syncthreads();

    // Reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(count, sdata[0]);
    }
}

// ============================================================================
// Host Functions
// ============================================================================

void relu_forward(const float* input, float* output, int size) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    relu_forward_kernel<<<gridSize, blockSize>>>(input, output, size);
    CUDA_CHECK(cudaGetLastError());
}

void relu_backward(const float* grad_output, const float* input,
                   float* grad_input, int size) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    relu_backward_kernel<<<gridSize, blockSize>>>(
        grad_output, input, grad_input, size
    );
    CUDA_CHECK(cudaGetLastError());
}

void gelu_forward(const float* input, float* output, int size) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    gelu_forward_kernel<<<gridSize, blockSize>>>(input, output, size);
    CUDA_CHECK(cudaGetLastError());
}

void gelu_backward(const float* grad_output, const float* input,
                   float* grad_input, int size) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    gelu_backward_kernel<<<gridSize, blockSize>>>(
        grad_output, input, grad_input, size
    );
    CUDA_CHECK(cudaGetLastError());
}

void elementwise_add(const float* A, const float* B, float* C, int size) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    elementwise_add_kernel<<<gridSize, blockSize>>>(A, B, C, size);
    CUDA_CHECK(cudaGetLastError());
}

void elementwise_multiply(const float* A, const float* B, float* C, int size) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    elementwise_multiply_kernel<<<gridSize, blockSize>>>(A, B, C, size);
    CUDA_CHECK(cudaGetLastError());
}

void scale(float* A, float scale_factor, int size) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    scale_kernel<<<gridSize, blockSize>>>(A, scale_factor, size);
    CUDA_CHECK(cudaGetLastError());
}

void add_bias(const float* input, const float* bias, float* output,
              int batch_size, int hidden_dim) {
    dim3 blockDim(16, 16);
    dim3 gridDim(
        (hidden_dim + blockDim.x - 1) / blockDim.x,
        (batch_size + blockDim.y - 1) / blockDim.y
    );

    add_bias_kernel<<<gridDim, blockDim>>>(
        input, bias, output, batch_size, hidden_dim
    );
    CUDA_CHECK(cudaGetLastError());
}

void residual_add(const float* input, const float* residual, float* output,
                  float scale_factor, int size) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    residual_add_kernel<<<gridSize, blockSize>>>(
        input, residual, output, scale_factor, size
    );
    CUDA_CHECK(cudaGetLastError());
}

int check_finite(const float* input, int size) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    size_t sharedMemSize = blockSize * sizeof(int);

    int* d_count;
    CUDA_CHECK(cudaMalloc(&d_count, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_count, 0, sizeof(int)));

    check_finite_kernel<<<gridSize, blockSize, sharedMemSize>>>(
        input, d_count, size
    );
    CUDA_CHECK(cudaGetLastError());

    int h_count = 0;
    CUDA_CHECK(cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_count));

    return h_count;
}

} // namespace kernels
} // namespace tiny_transformer
