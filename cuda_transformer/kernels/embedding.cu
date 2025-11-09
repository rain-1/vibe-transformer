/**
 * Embedding kernels - forward and backward passes
 * Adapted from MLP-cuda reference implementation
 */

#include "../include/kernels.h"
#include <cuda_runtime.h>
#include <stdio.h>

namespace tiny_transformer {
namespace kernels {

/**
 * Embedding lookup forward kernel
 *
 * Given token IDs, looks up embeddings from embedding table
 *
 * token_ids:  [batch_size, seq_len] - integer token IDs
 * embeddings: [vocab_size, d_model] - embedding lookup table
 * output:     [batch_size, seq_len, d_model] - output embeddings
 */
__global__ void embedding_forward_kernel(
    const int* __restrict__ token_ids,
    const float* __restrict__ embeddings,
    float* __restrict__ output,
    int batch_size,
    int seq_len,
    int vocab_size,
    int d_model
) {
    int batch = blockIdx.y;
    int seq = blockIdx.x;
    int dim = threadIdx.x;

    if (batch < batch_size && seq < seq_len && dim < d_model) {
        int token_id = token_ids[batch * seq_len + seq];

        if (token_id >= 0 && token_id < vocab_size) {
            // Look up embedding
            int out_idx = batch * seq_len * d_model + seq * d_model + dim;
            int emb_idx = token_id * d_model + dim;
            output[out_idx] = embeddings[emb_idx];
        } else {
            // Out of range token ID - zero embedding
            int out_idx = batch * seq_len * d_model + seq * d_model + dim;
            output[out_idx] = 0.0f;
        }
    }
}

/**
 * Embedding backward kernel
 *
 * Accumulates gradients into the embedding table
 *
 * grad_output:     [batch_size, seq_len, d_model] - gradient from next layer
 * token_ids:       [batch_size, seq_len] - token IDs used in forward pass
 * grad_embeddings: [vocab_size, d_model] - gradient for embedding table (accumulated)
 *
 * Note: Uses atomicAdd since multiple positions may use the same token
 */
__global__ void embedding_backward_kernel(
    const float* __restrict__ grad_output,
    const int* __restrict__ token_ids,
    float* __restrict__ grad_embeddings,
    int batch_size,
    int seq_len,
    int vocab_size,
    int d_model
) {
    int batch = blockIdx.y;
    int seq = blockIdx.x;
    int dim = threadIdx.x;

    if (batch < batch_size && seq < seq_len && dim < d_model) {
        int token_id = token_ids[batch * seq_len + seq];

        if (token_id >= 0 && token_id < vocab_size) {
            // Accumulate gradient for this token's embedding
            int grad_out_idx = batch * seq_len * d_model + seq * d_model + dim;
            int grad_emb_idx = token_id * d_model + dim;

            float grad = grad_output[grad_out_idx];
            atomicAdd(&grad_embeddings[grad_emb_idx], grad);
        }
    }
}

/**
 * Sinusoidal position encoding kernel
 *
 * Computes sinusoidal position encodings:
 * PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
 * PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
 *
 * output: [max_seq_len, d_model]
 */
__global__ void sinusoidal_position_encoding_kernel(
    float* __restrict__ output,
    int max_seq_len,
    int d_model
) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (pos < max_seq_len && i < d_model / 2) {
        float position = (float)pos;
        float div_term = expf(2.0f * i * (-logf(10000.0f) / d_model));

        int idx_sin = pos * d_model + 2 * i;
        int idx_cos = pos * d_model + 2 * i + 1;

        output[idx_sin] = sinf(position * div_term);
        output[idx_cos] = cosf(position * div_term);
    }
}

/**
 * Add position embeddings to token embeddings
 *
 * token_embeddings: [batch_size, seq_len, d_model]
 * pos_embeddings:   [seq_len, d_model] or [max_seq_len, d_model]
 * output:           [batch_size, seq_len, d_model]
 */
__global__ void add_position_embeddings_kernel(
    const float* __restrict__ token_embeddings,
    const float* __restrict__ pos_embeddings,
    float* __restrict__ output,
    int batch_size,
    int seq_len,
    int d_model
) {
    int batch = blockIdx.z;
    int seq = blockIdx.y;
    int dim = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch < batch_size && seq < seq_len && dim < d_model) {
        int idx = batch * seq_len * d_model + seq * d_model + dim;
        int pos_idx = seq * d_model + dim;

        output[idx] = token_embeddings[idx] + pos_embeddings[pos_idx];
    }
}

// ============================================================================
// Host Functions
// ============================================================================

void embedding_forward(
    const int* token_ids,
    const float* embeddings,
    float* output,
    int batch_size,
    int seq_len,
    int vocab_size,
    int d_model
) {
    // Use 2D grid for batch and sequence dimensions
    // Use 1D blocks for embedding dimension
    dim3 blockDim(d_model);
    dim3 gridDim(seq_len, batch_size);

    embedding_forward_kernel<<<gridDim, blockDim>>>(
        token_ids, embeddings, output,
        batch_size, seq_len, vocab_size, d_model
    );
    CUDA_CHECK(cudaGetLastError());
}

void embedding_backward(
    const float* grad_output,
    const int* token_ids,
    float* grad_embeddings,
    int batch_size,
    int seq_len,
    int vocab_size,
    int d_model
) {
    // Zero out gradient buffer first
    CUDA_CHECK(cudaMemset(grad_embeddings, 0, vocab_size * d_model * sizeof(float)));

    dim3 blockDim(d_model);
    dim3 gridDim(seq_len, batch_size);

    embedding_backward_kernel<<<gridDim, blockDim>>>(
        grad_output, token_ids, grad_embeddings,
        batch_size, seq_len, vocab_size, d_model
    );
    CUDA_CHECK(cudaGetLastError());
}

void create_sinusoidal_position_encodings(
    float* output,
    int max_seq_len,
    int d_model
) {
    dim3 blockDim(16, 16);
    dim3 gridDim(
        (max_seq_len + blockDim.x - 1) / blockDim.x,
        ((d_model / 2) + blockDim.y - 1) / blockDim.y
    );

    sinusoidal_position_encoding_kernel<<<gridDim, blockDim>>>(
        output, max_seq_len, d_model
    );
    CUDA_CHECK(cudaGetLastError());
}

void add_position_embeddings(
    const float* token_embeddings,
    const float* pos_embeddings,
    float* output,
    int batch_size,
    int seq_len,
    int d_model
) {
    int threads_per_block = 256;
    int blocks_for_dim = (d_model + threads_per_block - 1) / threads_per_block;

    dim3 blockDim(threads_per_block);
    dim3 gridDim(blocks_for_dim, seq_len, batch_size);

    add_position_embeddings_kernel<<<gridDim, blockDim>>>(
        token_embeddings, pos_embeddings, output,
        batch_size, seq_len, d_model
    );
    CUDA_CHECK(cudaGetLastError());
}

} // namespace kernels
} // namespace tiny_transformer
