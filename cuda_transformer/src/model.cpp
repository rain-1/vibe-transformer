/**
 * TinyTransformer model implementation
 */

#include "../include/model.h"
#include "../include/kernels.h"
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>

namespace tiny_transformer {

TinyTransformer::TinyTransformer(
    int vocab_size,
    int d_model,
    int n_heads,
    int d_ff,
    int max_seq_len,
    int n_blocks,
    float dropout,
    bool use_rms_norm,
    bool use_sinusoidal_pos,
    float residual_scale
) : vocab_size_(vocab_size),
    d_model_(d_model),
    n_heads_(n_heads),
    d_ff_(d_ff),
    max_seq_len_(max_seq_len),
    n_blocks_(n_blocks),
    dropout_(dropout),
    use_rms_norm_(use_rms_norm),
    use_sinusoidal_pos_(use_sinusoidal_pos) {

    // Auto-compute residual scale if not provided
    // Standard practice: scale by 1/sqrt(2*n_blocks) for gradient stability
    if (residual_scale < 0.0f) {
        residual_scale_ = 1.0f / sqrtf(2.0f * n_blocks);
    } else {
        residual_scale_ = residual_scale;
    }

    // Create token embedding
    token_embedding_ = std::make_unique<Embedding>(vocab_size, d_model);

    // Create position encoding
    if (use_sinusoidal_pos) {
        sinusoidal_pos_ = std::make_unique<SinusoidalPositionEncoding>(max_seq_len, d_model);
    } else {
        learned_pos_embedding_ = std::make_unique<Embedding>(max_seq_len, d_model);
    }

    // Create transformer blocks
    for (int i = 0; i < n_blocks; i++) {
        blocks_.push_back(std::make_unique<TransformerBlock>(
            d_model, n_heads, d_ff, dropout, use_rms_norm, residual_scale_
        ));
    }

    // Create final normalization layer
    if (use_rms_norm) {
        final_norm_ = std::make_unique<RMSNorm>(d_model);
    } else {
        final_norm_ = std::make_unique<LayerNorm>(d_model);
    }

    // Create output projection
    output_projection_ = std::make_unique<Linear>(d_model, vocab_size, true);

    // Initialize output projection weights
    float std = 1.0f / sqrtf(static_cast<float>(d_model));
    output_projection_->initialize(std);

    std::cout << "TinyTransformer initialized with " << count_parameters()
              << " parameters" << std::endl;
}

Tensor TinyTransformer::forward(
    const Tensor& input,
    const Tensor* mask,
    bool training
) {
    // input: [batch_size, seq_len] - token indices (stored as floats, interpreted as ints)
    int batch_size = input.shape()[0];
    int seq_len = input.shape()[1];

    // Allocate intermediate tensors if needed
    if (token_emb_ == nullptr ||
        token_emb_->shape()[0] != batch_size ||
        token_emb_->shape()[1] != seq_len) {
        token_emb_ = std::make_unique<Tensor>(
            std::vector<int>{batch_size, seq_len, d_model_}, false
        );
        pos_emb_ = std::make_unique<Tensor>(
            std::vector<int>{batch_size, seq_len, d_model_}, false
        );
        combined_emb_ = std::make_unique<Tensor>(
            std::vector<int>{batch_size, seq_len, d_model_}, false
        );
        final_normed_ = std::make_unique<Tensor>(
            std::vector<int>{batch_size, seq_len, d_model_}, false
        );
    }

    // 1. Token embedding
    token_embedding_->forward(input, *token_emb_, training);

    // 2. Position encoding
    if (use_sinusoidal_pos_) {
        // Sinusoidal position encodings
        sinusoidal_pos_->forward(*pos_emb_, seq_len);
    } else {
        // Learned position embeddings
        // Create position indices: [0, 1, 2, ..., seq_len-1] for each batch
        std::vector<float> pos_indices_host(batch_size * seq_len);
        for (int b = 0; b < batch_size; b++) {
            for (int s = 0; s < seq_len; s++) {
                pos_indices_host[b * seq_len + s] = static_cast<float>(s);
            }
        }

        Tensor pos_indices(std::vector<int>{batch_size, seq_len}, false);
        pos_indices.copy_from_host(pos_indices_host.data());

        learned_pos_embedding_->forward(pos_indices, *pos_emb_, training);
    }

    // 3. Combine embeddings: combined = token_emb + pos_emb
    CUDA_CHECK(cudaMemcpy(
        combined_emb_->data(),
        token_emb_->data(),
        token_emb_->numel() * sizeof(float),
        cudaMemcpyDeviceToDevice
    ));
    kernels::elementwise_add(
        combined_emb_->data(),
        pos_emb_->data(),
        combined_emb_->data(),
        combined_emb_->numel()
    );

    // 4. Apply transformer blocks
    // We'll ping-pong between two buffers to avoid unnecessary copies
    block_outputs_.clear();
    Tensor* current = combined_emb_.get();

    for (size_t i = 0; i < blocks_.size(); i++) {
        auto block_out = std::make_unique<Tensor>(
            std::vector<int>{batch_size, seq_len, d_model_}, false
        );

        blocks_[i]->forward(*current, *block_out, mask, false, training);

        current = block_out.get();
        block_outputs_.push_back(std::move(block_out));
    }

    // 5. Final layer norm
    if (auto* ln = dynamic_cast<LayerNorm*>(final_norm_.get())) {
        ln->forward(*current, *final_normed_, training);
    } else if (auto* rn = dynamic_cast<RMSNorm*>(final_norm_.get())) {
        rn->forward(*current, *final_normed_, training);
    }

    // 6. Output projection to vocabulary
    // final_normed: [batch_size, seq_len, d_model]
    // Need to flatten to [batch_size * seq_len, d_model] for linear layer
    int flat_batch = batch_size * seq_len;
    Tensor flat_input(final_normed_->data(),
                      std::vector<int>{flat_batch, d_model_}, false);

    Tensor logits(std::vector<int>{batch_size, seq_len, vocab_size_}, true);
    Tensor flat_logits(logits.data(),
                       std::vector<int>{flat_batch, vocab_size_}, false);

    output_projection_->forward(flat_input, flat_logits, training);

    return logits;
}

void TinyTransformer::backward(const Tensor& grad_output, const Tensor& input) {
    // TODO: Implement full backward pass
    // This is very complex and requires:
    // 1. Backward through output projection
    // 2. Backward through final norm
    // 3. Backward through each transformer block (in reverse order)
    // 4. Backward through embedding addition
    // 5. Backward through position encoding (if learned)
    // 6. Backward through token embedding

    // For now, we'll use PyTorch for training and only implement forward pass in CUDA
    // Full backward implementation would be needed for pure CUDA training
}

std::vector<Tensor*> TinyTransformer::parameters() {
    std::vector<Tensor*> params;

    // Token embedding parameters
    auto tok_params = token_embedding_->parameters();
    params.insert(params.end(), tok_params.begin(), tok_params.end());

    // Position embedding parameters (if learned)
    if (!use_sinusoidal_pos_) {
        auto pos_params = learned_pos_embedding_->parameters();
        params.insert(params.end(), pos_params.begin(), pos_params.end());
    }

    // Transformer block parameters
    for (auto& block : blocks_) {
        auto block_params = block->parameters();
        params.insert(params.end(), block_params.begin(), block_params.end());
    }

    // Final norm parameters
    auto norm_params = final_norm_->parameters();
    params.insert(params.end(), norm_params.begin(), norm_params.end());

    // Output projection parameters
    auto out_params = output_projection_->parameters();
    params.insert(params.end(), out_params.begin(), out_params.end());

    return params;
}

int TinyTransformer::count_parameters() const {
    int count = 0;

    // Token embeddings: vocab_size * d_model
    count += vocab_size_ * d_model_;

    // Position embeddings (if learned)
    if (!use_sinusoidal_pos_) {
        count += max_seq_len_ * d_model_;
    }

    // Each transformer block:
    // - 4 linear layers (Q, K, V, O) each d_model x d_model = 4 * d_model^2
    // - 2 FFN layers: d_model x d_ff + d_ff x d_model = d_model * d_ff * 2
    // - Biases: d_ff (linear1) + d_model (linear2) = d_ff + d_model
    // - Layer norms: 2 * d_model (gamma) + 2 * d_model (beta, if LayerNorm) = 2-4 * d_model
    int block_params = 4 * d_model_ * d_model_  // Attention projections
                     + 2 * d_model_ * d_ff_     // FFN
                     + d_ff_ + d_model_         // FFN biases
                     + (use_rms_norm_ ? 2 : 4) * d_model_;  // Norms

    count += n_blocks_ * block_params;

    // Final norm: d_model (gamma) + d_model (beta, if LayerNorm)
    count += use_rms_norm_ ? d_model_ : 2 * d_model_;

    // Output projection: d_model * vocab_size + vocab_size (bias)
    count += d_model_ * vocab_size_ + vocab_size_;

    return count;
}

// ============================================================================
// Utility Functions
// ============================================================================

std::unique_ptr<Tensor> create_causal_mask(int seq_len) {
    auto mask = std::make_unique<Tensor>(
        std::vector<int>{seq_len, seq_len}, false
    );

    kernels::create_causal_mask(mask->data(), seq_len);

    return mask;
}

} // namespace tiny_transformer
