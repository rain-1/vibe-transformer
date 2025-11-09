/**
 * Loss function kernels - cross-entropy loss for language modeling
 * Adapted from MLP-cuda reference implementation
 */

#include "../include/kernels.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

namespace tiny_transformer {
namespace kernels {

/**
 * Language modeling cross-entropy loss kernel
 *
 * Computes cross-entropy loss for language modeling: -log(softmax(logits)[target])
 * Uses numerically stable log-softmax computation
 *
 * logits:  [batch_size * seq_len, vocab_size] - raw model outputs
 * targets: [batch_size * seq_len] - target token IDs
 * mask:    [batch_size * seq_len] - 1 for valid positions, 0 for padding (optional)
 * partial_sums: [num_blocks] - partial loss sums for reduction
 *
 * For each position: loss = -log(softmax(logits)[target])
 *                         = -(logits[target] - log(Σ exp(logits)))
 */
__global__ void lm_cross_entropy_loss_kernel(
    const float* __restrict__ logits,
    const int* __restrict__ targets,
    const float* __restrict__ mask,
    float* __restrict__ partial_sums,
    int batch_size,
    int seq_len,
    int vocab_size
) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int total_positions = batch_size * seq_len;

    float local_loss = 0.0f;

    if (idx < total_positions) {
        // Check if this position is masked
        float m = (mask != nullptr) ? mask[idx] : 1.0f;

        if (m > 0.0f) {
            int target = targets[idx];

            if (target >= 0 && target < vocab_size) {
                // Get logits for this position
                const float* logits_ptr = logits + idx * vocab_size;

                // Compute max logit for numerical stability
                float max_logit = -INFINITY;
                for (int i = 0; i < vocab_size; i++) {
                    max_logit = fmaxf(max_logit, logits_ptr[i]);
                }

                // Compute log-sum-exp: log(Σ exp(logits - max))
                float sum_exp = 0.0f;
                for (int i = 0; i < vocab_size; i++) {
                    sum_exp += expf(logits_ptr[i] - max_logit);
                }
                float log_sum_exp = logf(sum_exp) + max_logit;

                // Cross-entropy: -log(softmax(logits)[target])
                //                = -(logits[target] - log_sum_exp)
                local_loss = -(logits_ptr[target] - log_sum_exp) * m;
            }
        }
    }

    sdata[tid] = local_loss;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

/**
 * Language modeling cross-entropy gradient kernel
 *
 * Computes gradient: softmax(logits) - one_hot(target), scaled by mask
 *
 * For cross-entropy loss: L = -log(softmax(logits)[target])
 * Gradient: ∂L/∂logits = softmax(logits) - one_hot(target)
 *
 * This kernel computes gradients for vocabularies of any size using grid-stride loops
 */
__global__ void lm_cross_entropy_gradient_kernel(
    const float* __restrict__ logits,
    const int* __restrict__ targets,
    const float* __restrict__ mask,
    float* __restrict__ grad,
    int batch_size,
    int seq_len,
    int vocab_size,
    float scale
) {
    int idx = blockIdx.x;  // Position index (batch * seq_len)
    int total_positions = batch_size * seq_len;

    if (idx >= total_positions) return;

    // Check if this position is masked
    float m = (mask != nullptr) ? mask[idx] : 1.0f;

    const float* logits_ptr = logits + idx * vocab_size;
    float* grad_ptr = grad + idx * vocab_size;

    if (m > 0.0f) {
        int target = targets[idx];

        // Use shared memory for reduction
        extern __shared__ float sdata[];
        float* s_max = sdata;
        float* s_sum = sdata + blockDim.x;

        // Find max logit using reduction
        float thread_max = -INFINITY;
        for (int v = threadIdx.x; v < vocab_size; v += blockDim.x) {
            thread_max = fmaxf(thread_max, logits_ptr[v]);
        }
        s_max[threadIdx.x] = thread_max;
        __syncthreads();

        // Reduce to find global max
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                s_max[threadIdx.x] = fmaxf(s_max[threadIdx.x], s_max[threadIdx.x + s]);
            }
            __syncthreads();
        }
        float max_logit = s_max[0];
        __syncthreads();

        // Compute sum of exp using reduction
        float thread_sum = 0.0f;
        for (int v = threadIdx.x; v < vocab_size; v += blockDim.x) {
            thread_sum += expf(logits_ptr[v] - max_logit);
        }
        s_sum[threadIdx.x] = thread_sum;
        __syncthreads();

        // Reduce to find global sum
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                s_sum[threadIdx.x] += s_sum[threadIdx.x + s];
            }
            __syncthreads();
        }
        float sum_exp = s_sum[0];
        __syncthreads();

        // Compute gradient for all vocabulary items this thread is responsible for
        for (int v = threadIdx.x; v < vocab_size; v += blockDim.x) {
            float softmax_v = expf(logits_ptr[v] - max_logit) / sum_exp;
            float target_indicator = (v == target) ? 1.0f : 0.0f;
            grad_ptr[v] = scale * m * (softmax_v - target_indicator);
        }
    } else {
        // Masked position - zero gradient
        for (int v = threadIdx.x; v < vocab_size; v += blockDim.x) {
            grad_ptr[v] = 0.0f;
        }
    }
}

/**
 * Compute accuracy kernel
 *
 * Computes fraction of correct predictions
 */
__global__ void compute_accuracy_kernel(
    const float* __restrict__ logits,
    const int* __restrict__ targets,
    const float* __restrict__ mask,
    int* __restrict__ partial_correct,
    int* __restrict__ partial_total,
    int batch_size,
    int seq_len,
    int vocab_size
) {
    extern __shared__ int isdata[];
    int* s_correct = isdata;
    int* s_total = isdata + blockDim.x;

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int total_positions = batch_size * seq_len;

    int local_correct = 0;
    int local_total = 0;

    if (idx < total_positions) {
        float m = (mask != nullptr) ? mask[idx] : 1.0f;

        if (m > 0.0f) {
            local_total = 1;

            int target = targets[idx];
            const float* logits_ptr = logits + idx * vocab_size;

            // Find argmax
            int pred = 0;
            float max_val = logits_ptr[0];
            for (int i = 1; i < vocab_size; i++) {
                if (logits_ptr[i] > max_val) {
                    max_val = logits_ptr[i];
                    pred = i;
                }
            }

            if (pred == target) {
                local_correct = 1;
            }
        }
    }

    s_correct[tid] = local_correct;
    s_total[tid] = local_total;
    __syncthreads();

    // Reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_correct[tid] += s_correct[tid + s];
            s_total[tid] += s_total[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_correct[blockIdx.x] = s_correct[0];
        partial_total[blockIdx.x] = s_total[0];
    }
}

// ============================================================================
// Host Functions
// ============================================================================

float lm_cross_entropy_loss(
    const float* logits,
    const int* targets,
    int batch_size,
    int seq_len,
    int vocab_size,
    const float* mask
) {
    int total_positions = batch_size * seq_len;
    int blockSize = 256;
    int gridSize = (total_positions + blockSize - 1) / blockSize;
    size_t sharedMemSize = blockSize * sizeof(float);

    // Allocate device memory for partial sums
    float* d_partial_sums;
    CUDA_CHECK(cudaMalloc(&d_partial_sums, gridSize * sizeof(float)));

    // Compute partial sums
    lm_cross_entropy_loss_kernel<<<gridSize, blockSize, sharedMemSize>>>(
        logits, targets, mask, d_partial_sums,
        batch_size, seq_len, vocab_size
    );
    CUDA_CHECK(cudaGetLastError());

    // Copy partial sums to host and reduce
    float* h_partial_sums = new float[gridSize];
    CUDA_CHECK(cudaMemcpy(h_partial_sums, d_partial_sums, gridSize * sizeof(float),
                         cudaMemcpyDeviceToHost));

    float total_loss = 0.0f;
    for (int i = 0; i < gridSize; i++) {
        total_loss += h_partial_sums[i];
    }

    // Cleanup
    delete[] h_partial_sums;
    CUDA_CHECK(cudaFree(d_partial_sums));

    // Compute number of non-masked positions
    int num_valid = total_positions;
    if (mask != nullptr) {
        float* h_mask = new float[total_positions];
        CUDA_CHECK(cudaMemcpy(h_mask, mask, total_positions * sizeof(float),
                             cudaMemcpyDeviceToHost));
        num_valid = 0;
        for (int i = 0; i < total_positions; i++) {
            if (h_mask[i] > 0.0f) num_valid++;
        }
        delete[] h_mask;
    }

    return (num_valid > 0) ? (total_loss / num_valid) : 0.0f;
}

void lm_cross_entropy_gradient(
    const float* logits,
    const int* targets,
    float* grad,
    int batch_size,
    int seq_len,
    int vocab_size,
    const float* mask
) {
    int total_positions = batch_size * seq_len;

    // Compute number of non-masked positions for scaling
    int num_valid = total_positions;
    if (mask != nullptr) {
        float* h_mask = new float[total_positions];
        CUDA_CHECK(cudaMemcpy(h_mask, mask, total_positions * sizeof(float),
                             cudaMemcpyDeviceToHost));
        num_valid = 0;
        for (int i = 0; i < total_positions; i++) {
            if (h_mask[i] > 0.0f) num_valid++;
        }
        delete[] h_mask;
    }

    float scale = (num_valid > 0) ? (1.0f / num_valid) : 0.0f;

    // Launch kernel: one block per position, fixed block size
    int blockSize = 256;
    dim3 gridSize(total_positions);
    dim3 threads(blockSize);
    size_t sharedMemSize = 2 * blockSize * sizeof(float);

    lm_cross_entropy_gradient_kernel<<<gridSize, threads, sharedMemSize>>>(
        logits, targets, mask, grad,
        batch_size, seq_len, vocab_size, scale
    );
    CUDA_CHECK(cudaGetLastError());
}

float compute_accuracy(
    const float* logits,
    const int* targets,
    int batch_size,
    int seq_len,
    int vocab_size,
    const float* mask
) {
    int total_positions = batch_size * seq_len;
    int blockSize = 256;
    int gridSize = (total_positions + blockSize - 1) / blockSize;
    size_t sharedMemSize = 2 * blockSize * sizeof(int);

    // Allocate device memory
    int* d_partial_correct;
    int* d_partial_total;
    CUDA_CHECK(cudaMalloc(&d_partial_correct, gridSize * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_partial_total, gridSize * sizeof(int)));

    // Compute partial counts
    compute_accuracy_kernel<<<gridSize, blockSize, sharedMemSize>>>(
        logits, targets, mask,
        d_partial_correct, d_partial_total,
        batch_size, seq_len, vocab_size
    );
    CUDA_CHECK(cudaGetLastError());

    // Copy to host and reduce
    int* h_partial_correct = new int[gridSize];
    int* h_partial_total = new int[gridSize];
    CUDA_CHECK(cudaMemcpy(h_partial_correct, d_partial_correct, gridSize * sizeof(int),
                         cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_partial_total, d_partial_total, gridSize * sizeof(int),
                         cudaMemcpyDeviceToHost));

    int total_correct = 0;
    int total_count = 0;
    for (int i = 0; i < gridSize; i++) {
        total_correct += h_partial_correct[i];
        total_count += h_partial_total[i];
    }

    // Cleanup
    delete[] h_partial_correct;
    delete[] h_partial_total;
    CUDA_CHECK(cudaFree(d_partial_correct));
    CUDA_CHECK(cudaFree(d_partial_total));

    return (total_count > 0) ? ((float)total_correct / total_count) : 0.0f;
}

} // namespace kernels
} // namespace tiny_transformer
