/**
 * High-level layer implementations for transformer.
 * Each layer manages its own parameters and forward/backward passes.
 */

#pragma once

#include "tensor.h"
#include <memory>
#include <vector>

namespace tiny_transformer {

// ============================================================================
// Base Layer
// ============================================================================

class Layer {
public:
    virtual ~Layer() = default;

    virtual std::vector<Tensor*> parameters() = 0;
    virtual std::string name() const = 0;
};

// ============================================================================
// Embedding Layer
// ============================================================================

class Embedding : public Layer {
public:
    Embedding(int vocab_size, int embedding_dim);

    void forward(const Tensor& input, Tensor& output, bool training = true);
    void backward(const Tensor& grad_output, const Tensor& input);

    std::vector<Tensor*> parameters() override;
    std::string name() const override { return "Embedding"; }

    Tensor& weight() { return *weight_; }

private:
    std::unique_ptr<Tensor> weight_;  // [vocab_size, embedding_dim]
    int vocab_size_;
    int embedding_dim_;
};

// ============================================================================
// Position Encoding
// ============================================================================

class SinusoidalPositionEncoding {
public:
    SinusoidalPositionEncoding(int max_seq_len, int d_model);

    void forward(Tensor& output, int seq_len);  // No backward needed (fixed)

    const Tensor& encoding() const { return *encoding_; }

private:
    std::unique_ptr<Tensor> encoding_;  // [max_seq_len, d_model]
    int max_seq_len_;
    int d_model_;
};

// ============================================================================
// LayerNorm
// ============================================================================

class LayerNorm : public Layer {
public:
    LayerNorm(int hidden_dim, float eps = 1e-8f);

    void forward(const Tensor& input, Tensor& output, bool training = true);
    void backward(const Tensor& grad_output, const Tensor& input);

    std::vector<Tensor*> parameters() override;
    std::string name() const override { return "LayerNorm"; }

private:
    std::unique_ptr<Tensor> gamma_;  // [hidden_dim]
    std::unique_ptr<Tensor> beta_;   // [hidden_dim]
    std::unique_ptr<Tensor> mean_;   // Saved for backward
    std::unique_ptr<Tensor> rstd_;   // Saved for backward
    int hidden_dim_;
    float eps_;
};

// ============================================================================
// RMSNorm
// ============================================================================

class RMSNorm : public Layer {
public:
    RMSNorm(int hidden_dim, float eps = 1e-8f);

    void forward(const Tensor& input, Tensor& output, bool training = true);
    void backward(const Tensor& grad_output, const Tensor& input);

    std::vector<Tensor*> parameters() override;
    std::string name() const override { return "RMSNorm"; }

private:
    std::unique_ptr<Tensor> gamma_;  // [hidden_dim]
    std::unique_ptr<Tensor> rms_;    // Saved for backward
    int hidden_dim_;
    float eps_;
};

// ============================================================================
// Linear Layer
// ============================================================================

class Linear : public Layer {
public:
    Linear(int in_features, int out_features, bool bias = true);

    void forward(const Tensor& input, Tensor& output, bool training = true);
    void backward(const Tensor& grad_output, const Tensor& input);

    std::vector<Tensor*> parameters() override;
    std::string name() const override { return "Linear"; }

    void initialize(float std);  // Xavier/Glorot initialization

private:
    std::unique_ptr<Tensor> weight_;  // [out_features, in_features]
    std::unique_ptr<Tensor> bias_;    // [out_features]
    int in_features_;
    int out_features_;
    bool use_bias_;
};

// ============================================================================
// Multi-Head Attention
// ============================================================================

class MultiHeadAttention : public Layer {
public:
    MultiHeadAttention(int d_model, int n_heads, float dropout = 0.0f);

    void forward(
        const Tensor& input, Tensor& output,
        const Tensor* mask = nullptr,
        bool return_attn_weights = false,
        bool training = true
    );

    void backward(const Tensor& grad_output, const Tensor& input);

    std::vector<Tensor*> parameters() override;
    std::string name() const override { return "MultiHeadAttention"; }

    const Tensor* attn_weights() const { return attn_weights_.get(); }

private:
    std::unique_ptr<Linear> W_q_;
    std::unique_ptr<Linear> W_k_;
    std::unique_ptr<Linear> W_v_;
    std::unique_ptr<Linear> W_o_;

    // Intermediate tensors (saved for backward)
    std::unique_ptr<Tensor> Q_, K_, V_;  // Flat projections
    std::unique_ptr<Tensor> Q_reshaped_, K_reshaped_, V_reshaped_;  // Reshaped for multi-head
    std::unique_ptr<Tensor> attn_scores_;
    std::unique_ptr<Tensor> attn_weights_;
    std::unique_ptr<Tensor> context_;
    std::unique_ptr<Tensor> context_concat_;  // Concatenated context before output projection

    int d_model_;
    int n_heads_;
    int d_k_;  // d_model / n_heads
    float dropout_;
    float scale_;  // 1 / sqrt(d_k)
};

// ============================================================================
// Feed-Forward Network
// ============================================================================

class FeedForward : public Layer {
public:
    FeedForward(int d_model, int d_ff, float dropout = 0.0f);

    void forward(const Tensor& input, Tensor& output, bool training = true);
    void backward(const Tensor& grad_output, const Tensor& input);

    std::vector<Tensor*> parameters() override;
    std::string name() const override { return "FeedForward"; }

private:
    std::unique_ptr<Linear> linear1_;
    std::unique_ptr<Linear> linear2_;
    std::unique_ptr<Tensor> hidden_;  // Saved for backward (post-activation)
    std::unique_ptr<Tensor> hidden_pre_act_;  // Saved for backward (pre-activation)
    float dropout_;
    int d_model_;
    int d_ff_;
};

// ============================================================================
// Transformer Block
// ============================================================================

class TransformerBlock : public Layer {
public:
    TransformerBlock(
        int d_model, int n_heads, int d_ff,
        float dropout = 0.0f, bool use_rms_norm = false,
        float residual_scale = 1.0f
    );

    void forward(
        const Tensor& input, Tensor& output,
        const Tensor* mask = nullptr,
        bool return_attn_weights = false,
        bool training = true
    );

    void backward(const Tensor& grad_output, const Tensor& input);

    std::vector<Tensor*> parameters() override;
    std::string name() const override { return "TransformerBlock"; }

private:
    std::unique_ptr<Layer> norm1_;  // LayerNorm or RMSNorm
    std::unique_ptr<Layer> norm2_;
    std::unique_ptr<MultiHeadAttention> attn_;
    std::unique_ptr<FeedForward> ffn_;

    // Intermediate tensors (saved for backward)
    std::unique_ptr<Tensor> normed1_;  // Output of norm1
    std::unique_ptr<Tensor> normed2_;  // Output of norm2
    std::unique_ptr<Tensor> attn_out_;
    std::unique_ptr<Tensor> ffn_out_;
    std::unique_ptr<Tensor> residual1_;
    std::unique_ptr<Tensor> residual2_;

    float residual_scale_;
};

} // namespace tiny_transformer
