/**
 * Attention kernels - scaled dot-product attention with forward and backward passes
 * Adapted from MLP-cuda reference implementation
 */

#include "../include/kernels.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

namespace tiny_transformer {
namespace kernels {

// Forward declarations of softmax functions from softmax.cu
void softmax_forward(const float* input, float* output,
                    int batch_size, int seq_len, int dim);
void softmax_backward(const float* output, const float* grad_output,
                     float* grad_input,
                     int batch_size, int seq_len, int dim);

// Forward declarations of matrix operations from matmul.cu
void matmul_forward(const float* A, const float* B, float* C,
                   int M, int N, int K, float alpha, float beta);
void matmul_backward(const float* A, const float* B, const float* dC,
                    float* dA, float* dB, int M, int N, int K);
void matmul_transB_custom(const float* A, const float* B, float* C,
                         int M, int K, int N);
void matmul_transA_custom(const float* A, const float* B, float* C,
                         int M, int K, int N);
void scale(float* A, float scale_factor, int size);

/**
 * Apply attention mask kernel
 * Sets masked positions to large negative value before softmax
 *
 * scores: [batch_size, seq_len, seq_len]
 * mask:   [batch_size, seq_len, seq_len] (1 = attend, 0 = mask out)
 */
__global__ void apply_attention_mask_kernel(
    float* __restrict__ scores,
    const float* __restrict__ mask,
    int batch_size,
    int seq_len
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * seq_len * seq_len;

    if (idx < total) {
        if (mask[idx] == 0.0f) {
            scores[idx] = -1e9f;  // Large negative value
        }
    }
}

/**
 * Create causal mask kernel
 * Creates lower triangular mask for autoregressive generation
 *
 * mask: [seq_len, seq_len]
 * mask[i][j] = 1 if i >= j else 0
 */
__global__ void create_causal_mask_kernel(
    float* __restrict__ mask,
    int seq_len
) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < seq_len && j < seq_len) {
        mask[i * seq_len + j] = (i >= j) ? 1.0f : 0.0f;
    }
}

/**
 * Reshape for multi-head attention
 * Converts [B, N, h*d] to [B*h, N, d]
 */
__global__ void reshape_for_attention_kernel(
    const float* __restrict__ input,   // [B, N, h*d]
    float* __restrict__ output,        // [B*h, N, d]
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim
) {
    int batch_head = blockIdx.z;  // 0 to B*h-1
    int b = batch_head / num_heads;
    int h = batch_head % num_heads;

    int n = blockIdx.y * blockDim.y + threadIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < seq_len && d < head_dim) {
        // Input: [b][n][h * head_dim + d]
        int in_idx = b * (seq_len * num_heads * head_dim) +
                     n * (num_heads * head_dim) +
                     h * head_dim + d;

        // Output: [batch_head][n][d]
        int out_idx = batch_head * (seq_len * head_dim) +
                      n * head_dim + d;

        output[out_idx] = input[in_idx];
    }
}

/**
 * Reshape from multi-head attention
 * Converts [B*h, N, d] back to [B, N, h*d]
 */
__global__ void reshape_from_attention_kernel(
    const float* __restrict__ input,   // [B*h, N, d]
    float* __restrict__ output,        // [B, N, h*d]
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim
) {
    int batch_head = blockIdx.z;  // 0 to B*h-1
    int b = batch_head / num_heads;
    int h = batch_head % num_heads;

    int n = blockIdx.y * blockDim.y + threadIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < seq_len && d < head_dim) {
        // Input: [batch_head][n][d]
        int in_idx = batch_head * (seq_len * head_dim) +
                     n * head_dim + d;

        // Output: [b][n][h * head_dim + d]
        int out_idx = b * (seq_len * num_heads * head_dim) +
                      n * (num_heads * head_dim) +
                      h * head_dim + d;

        output[out_idx] = input[in_idx];
    }
}

// ============================================================================
// Host Functions
// ============================================================================

void apply_attention_mask(
    float* scores,
    const float* mask,
    int batch_size,
    int seq_len
) {
    int total = batch_size * seq_len * seq_len;
    int blockSize = 256;
    int gridSize = (total + blockSize - 1) / blockSize;

    apply_attention_mask_kernel<<<gridSize, blockSize>>>(
        scores, mask, batch_size, seq_len
    );
    CUDA_CHECK(cudaGetLastError());
}

void create_causal_mask(
    float* mask,
    int seq_len
) {
    dim3 blockDim(16, 16);
    dim3 gridDim(
        (seq_len + blockDim.x - 1) / blockDim.x,
        (seq_len + blockDim.y - 1) / blockDim.y
    );

    create_causal_mask_kernel<<<gridDim, blockDim>>>(mask, seq_len);
    CUDA_CHECK(cudaGetLastError());
}

void reshape_for_attention(
    const float* input,
    float* output,
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim
) {
    dim3 blockDim(16, 16);
    dim3 gridDim(
        (head_dim + blockDim.x - 1) / blockDim.x,
        (seq_len + blockDim.y - 1) / blockDim.y,
        batch_size * num_heads
    );

    reshape_for_attention_kernel<<<gridDim, blockDim>>>(
        input, output, batch_size, seq_len, num_heads, head_dim
    );
    CUDA_CHECK(cudaGetLastError());
}

void reshape_from_attention(
    const float* input,
    float* output,
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim
) {
    dim3 blockDim(16, 16);
    dim3 gridDim(
        (head_dim + blockDim.x - 1) / blockDim.x,
        (seq_len + blockDim.y - 1) / blockDim.y,
        batch_size * num_heads
    );

    reshape_from_attention_kernel<<<gridDim, blockDim>>>(
        input, output, batch_size, seq_len, num_heads, head_dim
    );
    CUDA_CHECK(cudaGetLastError());
}

/**
 * Scaled dot-product attention forward pass
 *
 * Computes: Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
 *
 * Q, K, V:            [Bh, N, d] where Bh = batch_size * num_heads
 * scores_buffer:      [Bh, N, N] - workspace for attention scores
 * attn_weights_buffer: [Bh, N, N] - workspace for softmax output
 * output:             [Bh, N, d]
 * mask:               [Bh, N, N] or nullptr (optional)
 */
void attention_forward(
    const float* Q,
    const float* K,
    const float* V,
    float* output,
    float* scores_buffer,
    float* attn_weights_buffer,
    int batch_heads,
    int seq_len,
    int head_dim,
    const float* mask
) {
    float scale = 1.0f / sqrtf((float)head_dim);

    // 1. Compute attention scores: Scores = Q @ K^T
    //    Q: [Bh, N, d], K: [Bh, N, d]  (note: K will be transposed)
    //    Scores: [Bh, N, N]
    // We need to do batch_heads separate matmuls: Q[i] @ K[i]^T for each i
    for (int bh = 0; bh < batch_heads; bh++) {
        const float* Q_i = Q + bh * seq_len * head_dim;
        const float* K_i = K + bh * seq_len * head_dim;
        float* scores_i = scores_buffer + bh * seq_len * seq_len;

        // scores_i = Q_i @ K_i^T  ([N x d] @ [d x N] = [N x N])
        matmul_transB_custom(Q_i, K_i, scores_i, seq_len, head_dim, seq_len);
    }

    // 2. Scale scores
    scale(scores_buffer, scale, batch_heads * seq_len * seq_len);

    // 3. Apply mask if provided
    if (mask != nullptr) {
        apply_attention_mask(scores_buffer, mask, batch_heads, seq_len);
    }

    // 4. Apply softmax: Attn = softmax(Scores)
    // Softmax over last dimension (seq_len)
    softmax_forward(scores_buffer, attn_weights_buffer,
                   batch_heads, seq_len, seq_len);

    // 5. Compute output: Output = Attn @ V
    //    Attn: [Bh, N, N], V: [Bh, N, d]
    //    Output: [Bh, N, d]
    for (int bh = 0; bh < batch_heads; bh++) {
        const float* attn_i = attn_weights_buffer + bh * seq_len * seq_len;
        const float* V_i = V + bh * seq_len * head_dim;
        float* output_i = output + bh * seq_len * head_dim;

        // output_i = attn_i @ V_i  ([N x N] @ [N x d] = [N x d])
        float alpha = 1.0f, beta = 0.0f;
        matmul_forward(attn_i, V_i, output_i, seq_len, head_dim, seq_len, alpha, beta);
    }
}

/**
 * Scaled dot-product attention backward pass
 *
 * Given gradients w.r.t. output, computes gradients w.r.t. Q, K, V
 */
void attention_backward(
    const float* Q,
    const float* K,
    const float* V,
    const float* attn_weights,
    const float* grad_output,
    float* grad_Q,
    float* grad_K,
    float* grad_V,
    float* grad_scores_buffer,
    float* grad_attn_buffer,
    int batch_heads,
    int seq_len,
    int head_dim,
    const float* mask
) {
    float scale = 1.0f / sqrtf((float)head_dim);

    // Backward through: Output = Attn @ V
    for (int bh = 0; bh < batch_heads; bh++) {
        const float* attn_i = attn_weights + bh * seq_len * seq_len;
        const float* V_i = V + bh * seq_len * head_dim;
        const float* grad_out_i = grad_output + bh * seq_len * head_dim;
        float* grad_attn_i = grad_attn_buffer + bh * seq_len * seq_len;
        float* grad_V_i = grad_V + bh * seq_len * head_dim;

        matmul_backward(attn_i, V_i, grad_out_i,
                       grad_attn_i, grad_V_i,
                       seq_len, head_dim, seq_len);
    }

    // Backward through softmax
    softmax_backward(attn_weights, grad_attn_buffer, grad_scores_buffer,
                    batch_heads, seq_len, seq_len);

    // Backward through scaling
    scale(grad_scores_buffer, scale, batch_heads * seq_len * seq_len);

    // Note: Mask backward is not needed (mask is constant)

    // Backward through: Scores = Q @ K^T
    for (int bh = 0; bh < batch_heads; bh++) {
        const float* Q_i = Q + bh * seq_len * head_dim;
        const float* K_i = K + bh * seq_len * head_dim;
        const float* grad_scores_i = grad_scores_buffer + bh * seq_len * seq_len;
        float* grad_Q_i = grad_Q + bh * seq_len * head_dim;
        float* grad_K_i = grad_K + bh * seq_len * head_dim;

        // grad_Q = grad_scores @ K  ([N x N] @ [N x d] = [N x d])
        float alpha = 1.0f, beta = 0.0f;
        matmul_forward(grad_scores_i, K_i, grad_Q_i,
                      seq_len, head_dim, seq_len, alpha, beta);

        // grad_K = grad_scores^T @ Q  ([N x N]^T @ [N x d] = [N x d])
        matmul_transA_custom(grad_scores_i, Q_i, grad_K_i,
                            seq_len, seq_len, head_dim);
    }
}

} // namespace kernels
} // namespace tiny_transformer
