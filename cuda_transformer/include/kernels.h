/**
 * CUDA kernel declarations for all transformer operations.
 * Each operation includes both forward and backward passes.
 */

#pragma once

#include <cuda_runtime.h>
#include <stdio.h>

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

namespace tiny_transformer {
namespace kernels {

// ============================================================================
// Matrix Operations
// ============================================================================

// Matrix multiplication: C = alpha * A @ B + beta * C
// Forward: out = A @ B
// Backward: dA = dout @ B^T, dB = A^T @ dout
void matmul_forward(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    float alpha = 1.0f, float beta = 0.0f
);

void matmul_backward(
    const float* A, const float* B, const float* dC,
    float* dA, float* dB,
    int M, int N, int K
);

// ============================================================================
// Softmax
// ============================================================================

// Numerically stable softmax using log-sum-exp trick
// Forward: y_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
// Backward: dy/dx_i = y_i * (dL/dy_i - sum_j(y_j * dL/dy_j))
void softmax_forward(
    const float* input, float* output,
    int batch_size, int seq_len, int dim
);

void softmax_backward(
    const float* output, const float* grad_output, float* grad_input,
    int batch_size, int seq_len, int dim
);

// ============================================================================
// LayerNorm
// ============================================================================

// Layer normalization: y = (x - mean) / sqrt(var + eps) * gamma + beta
// Forward computes mean and var, applies normalization
// Backward: complex gradient flow through normalization
void layernorm_forward(
    const float* input, const float* gamma, const float* beta,
    float* output, float* mean, float* rstd,  // Save for backward
    int batch_size, int hidden_dim, float eps = 1e-8f
);

void layernorm_backward(
    const float* grad_output, const float* input,
    const float* gamma, const float* mean, const float* rstd,
    float* grad_input, float* grad_gamma, float* grad_beta,
    int batch_size, int hidden_dim
);

// ============================================================================
// RMSNorm
// ============================================================================

// RMS normalization: y = x / sqrt(mean(x^2) + eps) * gamma
// Simpler than LayerNorm, no bias term
void rmsnorm_forward(
    const float* input, const float* gamma,
    float* output, float* rms,  // Save RMS for backward
    int batch_size, int hidden_dim, float eps = 1e-8f
);

void rmsnorm_backward(
    const float* grad_output, const float* input,
    const float* gamma, const float* rms,
    float* grad_input, float* grad_gamma,
    int batch_size, int hidden_dim
);

// ============================================================================
// Attention
// ============================================================================

// Scaled dot-product attention
// Q, K, V: [batch, n_heads, seq_len, head_dim]
// Output: [batch, n_heads, seq_len, head_dim]
// Attention scores: [batch, n_heads, seq_len, seq_len]
void attention_forward(
    const float* Q, const float* K, const float* V,
    float* output, float* attn_scores,  // Save scores for backward
    int batch_size, int n_heads, int seq_len, int head_dim,
    float scale, const float* mask = nullptr
);

void attention_backward(
    const float* grad_output, const float* Q, const float* K, const float* V,
    const float* attn_scores,
    float* grad_Q, float* grad_K, float* grad_V,
    int batch_size, int n_heads, int seq_len, int head_dim,
    float scale
);

// ============================================================================
// Feed-Forward Network
// ============================================================================

// ReLU activation
void relu_forward(const float* input, float* output, int size);
void relu_backward(const float* grad_output, const float* input, float* grad_input, int size);

// GELU activation (optional, more commonly used in transformers)
void gelu_forward(const float* input, float* output, int size);
void gelu_backward(const float* grad_output, const float* input, float* grad_input, int size);

// ============================================================================
// Dropout
// ============================================================================

// Dropout: randomly zero out elements during training
void dropout_forward(
    const float* input, float* output, unsigned char* mask,
    int size, float p, bool training, unsigned long long seed
);

void dropout_backward(
    const float* grad_output, const unsigned char* mask,
    float* grad_input, int size, float p
);

// ============================================================================
// Embeddings
// ============================================================================

// Embedding lookup: output[i, :] = weight[input[i], :]
void embedding_forward(
    const int* input, const float* weight,
    float* output,
    int batch_size, int seq_len, int vocab_size, int embedding_dim
);

void embedding_backward(
    const int* input, const float* grad_output,
    float* grad_weight,
    int batch_size, int seq_len, int vocab_size, int embedding_dim
);

// Sinusoidal position encoding
void sinusoidal_position_encoding(
    float* output,
    int max_seq_len, int d_model
);

// ============================================================================
// Loss Functions
// ============================================================================

// Cross-entropy loss with softmax
// Input: [batch_size, vocab_size]
// Targets: [batch_size] (class indices)
void cross_entropy_forward(
    const float* logits, const int* targets,
    float* loss, float* softmax_output,  // Save softmax for backward
    int batch_size, int vocab_size
);

void cross_entropy_backward(
    const float* softmax_output, const int* targets,
    float* grad_logits,
    int batch_size, int vocab_size
);

// ============================================================================
// Utilities
// ============================================================================

// Element-wise operations
void add_inplace(float* a, const float* b, int size);
void mul_scalar(float* a, float scalar, int size);
void add_scaled(float* out, const float* a, const float* b, float scale, int size);

// Gradient clipping
float compute_global_norm(const float** grads, const int* sizes, int num_tensors);
void clip_grad_norm(float** grads, const int* sizes, int num_tensors, float max_norm);

// NaN/Inf checking
bool has_nan(const float* data, int size);
bool has_inf(const float* data, int size);

// Reductions
float reduce_sum(const float* data, int size);
float reduce_mean(const float* data, int size);
float reduce_max(const float* data, int size);
float reduce_norm(const float* data, int size);  // L2 norm

// Random initialization
void randn(float* data, int size, float mean, float std, unsigned long long seed);
void uniform(float* data, int size, float a, float b, unsigned long long seed);

} // namespace kernels
} // namespace tiny_transformer
