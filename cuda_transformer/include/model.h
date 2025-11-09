/**
 * TinyTransformer model - complete transformer for language modeling
 */

#pragma once

#include "tensor.h"
#include "layers.h"
#include <memory>
#include <vector>
#include <map>
#include <string>

namespace tiny_transformer {

/**
 * TinyTransformer - Complete autoregressive transformer model
 *
 * Architecture:
 *   1. Token embedding + position encoding
 *   2. N transformer blocks (pre-norm)
 *   3. Final layer norm
 *   4. Output projection to vocabulary
 */
class TinyTransformer {
public:
    TinyTransformer(
        int vocab_size,
        int d_model,
        int n_heads,
        int d_ff,
        int max_seq_len,
        int n_blocks = 1,
        float dropout = 0.1f,
        bool use_rms_norm = false,
        bool use_sinusoidal_pos = false,
        float residual_scale = -1.0f  // Auto-compute if negative
    );

    /**
     * Forward pass
     *
     * Args:
     *   input: [batch_size, seq_len] - token indices
     *   mask: [batch_size, seq_len, seq_len] - attention mask (optional)
     *   training: whether in training mode
     *
     * Returns:
     *   logits: [batch_size, seq_len, vocab_size]
     */
    Tensor forward(
        const Tensor& input,
        const Tensor* mask = nullptr,
        bool training = true
    );

    /**
     * Backward pass - compute gradients
     *
     * Args:
     *   grad_output: [batch_size, seq_len, vocab_size] - gradient from loss
     *   input: [batch_size, seq_len] - original input tokens
     */
    void backward(const Tensor& grad_output, const Tensor& input);

    /**
     * Get all trainable parameters
     */
    std::vector<Tensor*> parameters();

    /**
     * Get parameter count
     */
    int count_parameters() const;

    /**
     * Model configuration accessors
     */
    int vocab_size() const { return vocab_size_; }
    int d_model() const { return d_model_; }
    int n_heads() const { return n_heads_; }
    int n_blocks() const { return n_blocks_; }
    int max_seq_len() const { return max_seq_len_; }

private:
    // Model hyperparameters
    int vocab_size_;
    int d_model_;
    int n_heads_;
    int d_ff_;
    int max_seq_len_;
    int n_blocks_;
    float dropout_;
    bool use_rms_norm_;
    bool use_sinusoidal_pos_;
    float residual_scale_;

    // Model components
    std::unique_ptr<Embedding> token_embedding_;
    std::unique_ptr<SinusoidalPositionEncoding> sinusoidal_pos_;
    std::unique_ptr<Embedding> learned_pos_embedding_;  // If not using sinusoidal

    std::vector<std::unique_ptr<TransformerBlock>> blocks_;

    std::unique_ptr<Layer> final_norm_;  // LayerNorm or RMSNorm
    std::unique_ptr<Linear> output_projection_;

    // Intermediate tensors (saved for backward)
    std::unique_ptr<Tensor> token_emb_;
    std::unique_ptr<Tensor> pos_emb_;
    std::unique_ptr<Tensor> combined_emb_;
    std::vector<std::unique_ptr<Tensor>> block_outputs_;
    std::unique_ptr<Tensor> final_normed_;
};

/**
 * Create causal (autoregressive) attention mask
 *
 * Returns mask of shape [seq_len, seq_len] where mask[i][j] = 1 if i >= j else 0
 * This ensures position i can only attend to positions <= i
 */
std::unique_ptr<Tensor> create_causal_mask(int seq_len);

} // namespace tiny_transformer
