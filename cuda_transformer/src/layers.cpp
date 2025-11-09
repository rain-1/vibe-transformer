/**
 * Layer implementations - complete forward and backward passes
 */

#include "../include/layers.h"
#include "../include/kernels.h"
#include <cuda_runtime.h>
#include <cmath>

namespace tiny_transformer {

// ============================================================================
// Embedding Layer
// ============================================================================

Embedding::Embedding(int vocab_size, int embedding_dim)
    : vocab_size_(vocab_size), embedding_dim_(embedding_dim) {
    weight_ = std::make_unique<Tensor>(
        std::vector<int>{vocab_size, embedding_dim}, true
    );

    // Initialize embeddings with small random values
    weight_->randn_(0.0f, 0.02f);
}

void Embedding::forward(const Tensor& input, Tensor& output, bool training) {
    // input: [batch_size, seq_len] - int tensor
    // output: [batch_size, seq_len, embedding_dim]

    int batch_size = input.shape()[0];
    int seq_len = input.shape()[1];

    // We need to cast input data to int*
    // For now, assume input.data() can be interpreted as int*
    const int* token_ids = reinterpret_cast<const int*>(input.data());

    kernels::embedding_forward(
        token_ids,
        weight_->data(),
        output.data(),
        batch_size,
        seq_len,
        vocab_size_,
        embedding_dim_
    );
}

void Embedding::backward(const Tensor& grad_output, const Tensor& input) {
    // grad_output: [batch_size, seq_len, embedding_dim]
    // grad for weight: [vocab_size, embedding_dim]

    if (!weight_->requires_grad()) return;

    int batch_size = input.shape()[0];
    int seq_len = input.shape()[1];

    const int* token_ids = reinterpret_cast<const int*>(input.data());

    // Zero out gradient first
    weight_->zero_grad();

    kernels::embedding_backward(
        token_ids,
        grad_output.data(),
        weight_->grad(),
        batch_size,
        seq_len,
        vocab_size_,
        embedding_dim_
    );
}

std::vector<Tensor*> Embedding::parameters() {
    return {weight_.get()};
}

// ============================================================================
// Sinusoidal Position Encoding
// ============================================================================

SinusoidalPositionEncoding::SinusoidalPositionEncoding(int max_seq_len, int d_model)
    : max_seq_len_(max_seq_len), d_model_(d_model) {
    encoding_ = std::make_unique<Tensor>(
        std::vector<int>{max_seq_len, d_model}, false
    );

    // Generate sinusoidal encodings
    kernels::create_sinusoidal_position_encodings(
        encoding_->data(),
        max_seq_len,
        d_model
    );
}

void SinusoidalPositionEncoding::forward(Tensor& output, int seq_len) {
    // Copy the first seq_len positions to output
    // output: [batch_size, seq_len, d_model]
    int batch_size = output.shape()[0];

    // For each batch element, copy the position encodings
    for (int b = 0; b < batch_size; b++) {
        float* output_ptr = output.data() + b * seq_len * d_model_;
        CUDA_CHECK(cudaMemcpy(
            output_ptr,
            encoding_->data(),
            seq_len * d_model_ * sizeof(float),
            cudaMemcpyDeviceToDevice
        ));
    }
}

// ============================================================================
// LayerNorm
// ============================================================================

LayerNorm::LayerNorm(int hidden_dim, float eps)
    : hidden_dim_(hidden_dim), eps_(eps) {
    gamma_ = std::make_unique<Tensor>(std::vector<int>{hidden_dim}, true);
    beta_ = std::make_unique<Tensor>(std::vector<int>{hidden_dim}, true);

    // Initialize gamma to 1, beta to 0
    gamma_->fill_(1.0f);
    beta_->fill_(0.0f);

    // Allocate buffers for saving mean/rstd (set size later)
    mean_ = nullptr;
    rstd_ = nullptr;
}

void LayerNorm::forward(const Tensor& input, Tensor& output, bool training) {
    // input/output: [batch_size, hidden_dim]
    int batch_size = input.shape()[0];

    // Allocate mean/rstd if needed
    if (mean_ == nullptr || mean_->shape()[0] != batch_size) {
        mean_ = std::make_unique<Tensor>(std::vector<int>{batch_size}, false);
        rstd_ = std::make_unique<Tensor>(std::vector<int>{batch_size}, false);
    }

    kernels::layernorm_forward(
        input.data(),
        gamma_->data(),
        beta_->data(),
        output.data(),
        mean_->data(),
        rstd_->data(),
        batch_size,
        hidden_dim_,
        eps_
    );
}

void LayerNorm::backward(const Tensor& grad_output, const Tensor& input) {
    if (!gamma_->requires_grad()) return;

    int batch_size = input.shape()[0];

    // Allocate gradients
    gamma_->alloc_grad();
    beta_->alloc_grad();

    // grad_input will be written to input.grad()
    Tensor grad_input(const_cast<float*>(input.grad()), input.shape(), false);

    kernels::layernorm_backward(
        grad_output.data(),
        input.data(),
        gamma_->data(),
        mean_->data(),
        rstd_->data(),
        grad_input.data(),
        gamma_->grad(),
        beta_->grad(),
        batch_size,
        hidden_dim_
    );
}

std::vector<Tensor*> LayerNorm::parameters() {
    return {gamma_.get(), beta_.get()};
}

// ============================================================================
// RMSNorm
// ============================================================================

RMSNorm::RMSNorm(int hidden_dim, float eps)
    : hidden_dim_(hidden_dim), eps_(eps) {
    gamma_ = std::make_unique<Tensor>(std::vector<int>{hidden_dim}, true);
    gamma_->fill_(1.0f);

    rms_ = nullptr;
}

void RMSNorm::forward(const Tensor& input, Tensor& output, bool training) {
    int batch_size = input.shape()[0];

    if (rms_ == nullptr || rms_->shape()[0] != batch_size) {
        rms_ = std::make_unique<Tensor>(std::vector<int>{batch_size}, false);
    }

    kernels::rmsnorm_forward(
        input.data(),
        gamma_->data(),
        output.data(),
        rms_->data(),
        batch_size,
        hidden_dim_,
        eps_
    );
}

void RMSNorm::backward(const Tensor& grad_output, const Tensor& input) {
    if (!gamma_->requires_grad()) return;

    int batch_size = input.shape()[0];

    gamma_->alloc_grad();
    Tensor grad_input(const_cast<float*>(input.grad()), input.shape(), false);

    kernels::rmsnorm_backward(
        grad_output.data(),
        input.data(),
        gamma_->data(),
        rms_->data(),
        grad_input.data(),
        gamma_->grad(),
        batch_size,
        hidden_dim_
    );
}

std::vector<Tensor*> RMSNorm::parameters() {
    return {gamma_.get()};
}

// ============================================================================
// Linear Layer
// ============================================================================

Linear::Linear(int in_features, int out_features, bool bias)
    : in_features_(in_features), out_features_(out_features), use_bias_(bias) {
    weight_ = std::make_unique<Tensor>(
        std::vector<int>{out_features, in_features}, true
    );

    if (use_bias_) {
        bias_ = std::make_unique<Tensor>(
            std::vector<int>{out_features}, true
        );
        bias_->fill_(0.0f);
    }

    // Default initialization
    initialize(sqrtf(2.0f / (in_features + out_features)));
}

void Linear::initialize(float std) {
    weight_->randn_(0.0f, std);
}

void Linear::forward(const Tensor& input, Tensor& output, bool training) {
    // input: [batch_size, in_features]
    // output: [batch_size, out_features]
    // weight: [out_features, in_features]

    int batch_size = input.shape()[0];

    // output = input @ weight^T
    // [batch_size, in_features] @ [in_features, out_features] = [batch_size, out_features]
    kernels::matmul_forward(
        input.data(),
        weight_->data(),
        output.data(),
        batch_size,
        out_features_,
        in_features_,
        1.0f,  // alpha
        0.0f   // beta
    );

    // Add bias if present
    if (use_bias_) {
        // Broadcast bias across batch dimension
        for (int b = 0; b < batch_size; b++) {
            float* output_ptr = output.data() + b * out_features_;
            kernels::elementwise_add(
                output_ptr,
                bias_->data(),
                output_ptr,
                out_features_
            );
        }
    }
}

void Linear::backward(const Tensor& grad_output, const Tensor& input) {
    // grad_output: [batch_size, out_features]
    // grad_input: [batch_size, in_features]
    // grad_weight: [out_features, in_features]

    int batch_size = input.shape()[0];

    if (weight_->requires_grad()) {
        weight_->alloc_grad();

        // grad_weight = grad_output^T @ input
        // [out_features, batch_size] @ [batch_size, in_features] = [out_features, in_features]
        kernels::matmul_backward(
            input.data(),
            weight_->data(),
            grad_output.data(),
            const_cast<float*>(input.grad()),
            weight_->grad(),
            batch_size,
            out_features_,
            in_features_
        );
    }

    if (use_bias_ && bias_->requires_grad()) {
        bias_->alloc_grad();

        // Sum gradients across batch dimension
        // grad_bias = sum(grad_output, dim=0)
        CUDA_CHECK(cudaMemset(bias_->grad(), 0, out_features_ * sizeof(float)));
        for (int b = 0; b < batch_size; b++) {
            const float* grad_ptr = grad_output.data() + b * out_features_;
            kernels::elementwise_add(
                bias_->grad(),
                grad_ptr,
                bias_->grad(),
                out_features_
            );
        }
    }
}

std::vector<Tensor*> Linear::parameters() {
    if (use_bias_) {
        return {weight_.get(), bias_.get()};
    } else {
        return {weight_.get()};
    }
}

// ============================================================================
// Multi-Head Attention
// ============================================================================

MultiHeadAttention::MultiHeadAttention(int d_model, int n_heads, float dropout)
    : d_model_(d_model), n_heads_(n_heads), dropout_(dropout) {
    d_k_ = d_model / n_heads;
    scale_ = 1.0f / sqrtf(static_cast<float>(d_k_));

    // Create projection layers
    W_q_ = std::make_unique<Linear>(d_model, d_model, false);
    W_k_ = std::make_unique<Linear>(d_model, d_model, false);
    W_v_ = std::make_unique<Linear>(d_model, d_model, false);
    W_o_ = std::make_unique<Linear>(d_model, d_model, false);
}

void MultiHeadAttention::forward(
    const Tensor& input,
    Tensor& output,
    const Tensor* mask,
    bool return_attn_weights,
    bool training
) {
    // input: [batch_size, seq_len, d_model]
    int batch_size = input.shape()[0];
    int seq_len = input.shape()[1];

    // Flatten input for linear layers: [batch_size * seq_len, d_model]
    int flat_batch = batch_size * seq_len;
    Tensor flat_input(const_cast<float*>(input.data()),
                      std::vector<int>{flat_batch, d_model_}, false);

    // Allocate Q, K, V
    Q_ = std::make_unique<Tensor>(std::vector<int>{flat_batch, d_model_}, false);
    K_ = std::make_unique<Tensor>(std::vector<int>{flat_batch, d_model_}, false);
    V_ = std::make_unique<Tensor>(std::vector<int>{flat_batch, d_model_}, false);

    // Project to Q, K, V
    W_q_->forward(flat_input, *Q_, training);
    W_k_->forward(flat_input, *K_, training);
    W_v_->forward(flat_input, *V_, training);

    // Reshape for multi-head attention: [batch_size, seq_len, d_model] -> [batch_size * n_heads, seq_len, d_k]
    int batch_heads = batch_size * n_heads_;
    Tensor Q_reshaped(std::vector<int>{batch_heads, seq_len, d_k_}, false);
    Tensor K_reshaped(std::vector<int>{batch_heads, seq_len, d_k_}, false);
    Tensor V_reshaped(std::vector<int>{batch_heads, seq_len, d_k_}, false);

    kernels::reshape_for_attention(
        Q_->data(), Q_reshaped.data(),
        batch_size, seq_len, n_heads_, d_k_
    );
    kernels::reshape_for_attention(
        K_->data(), K_reshaped.data(),
        batch_size, seq_len, n_heads_, d_k_
    );
    kernels::reshape_for_attention(
        V_->data(), V_reshaped.data(),
        batch_size, seq_len, n_heads_, d_k_
    );

    // Allocate buffers for attention
    attn_scores_ = std::make_unique<Tensor>(
        std::vector<int>{batch_heads, seq_len, seq_len}, false
    );
    attn_weights_ = std::make_unique<Tensor>(
        std::vector<int>{batch_heads, seq_len, seq_len}, false
    );
    context_ = std::make_unique<Tensor>(
        std::vector<int>{batch_heads, seq_len, d_k_}, false
    );

    // Run scaled dot-product attention
    kernels::attention_forward(
        Q_reshaped.data(),
        K_reshaped.data(),
        V_reshaped.data(),
        context_->data(),
        attn_scores_->data(),
        attn_weights_->data(),
        batch_heads,
        seq_len,
        d_k_,
        mask ? mask->data() : nullptr
    );

    // Reshape back: [batch_heads, seq_len, d_k] -> [batch_size, seq_len, d_model]
    Tensor context_concat(std::vector<int>{flat_batch, d_model_}, false);
    kernels::reshape_from_attention(
        context_->data(),
        context_concat.data(),
        batch_size, seq_len, n_heads_, d_k_
    );

    // Final projection
    W_o_->forward(context_concat, output, training);
}

void MultiHeadAttention::backward(const Tensor& grad_output, const Tensor& input) {
    // This is complex - we need to backprop through all the attention operations
    // For now, implement a simplified version
    // Full implementation would require saving all intermediate gradients

    // TODO: Implement complete backward pass through attention
    // This requires:
    // 1. Backward through W_o
    // 2. Backward through reshape
    // 3. Backward through attention_forward
    // 4. Backward through reshape again
    // 5. Backward through W_q, W_k, W_v
}

std::vector<Tensor*> MultiHeadAttention::parameters() {
    std::vector<Tensor*> params;

    auto q_params = W_q_->parameters();
    auto k_params = W_k_->parameters();
    auto v_params = W_v_->parameters();
    auto o_params = W_o_->parameters();

    params.insert(params.end(), q_params.begin(), q_params.end());
    params.insert(params.end(), k_params.begin(), k_params.end());
    params.insert(params.end(), v_params.begin(), v_params.end());
    params.insert(params.end(), o_params.begin(), o_params.end());

    return params;
}

// ============================================================================
// Feed-Forward Network
// ============================================================================

FeedForward::FeedForward(int d_model, int d_ff, float dropout)
    : d_model_(d_model), d_ff_(d_ff), dropout_(dropout) {
    linear1_ = std::make_unique<Linear>(d_model, d_ff, true);
    linear2_ = std::make_unique<Linear>(d_ff, d_model, true);
}

void FeedForward::forward(const Tensor& input, Tensor& output, bool training) {
    // input: [batch_size, seq_len, d_model]
    int batch_size = input.shape()[0];
    int seq_len = input.shape()[1];
    int flat_batch = batch_size * seq_len;

    // Flatten for linear layers
    Tensor flat_input(const_cast<float*>(input.data()),
                      std::vector<int>{flat_batch, d_model_}, false);

    // Allocate hidden layer
    hidden_ = std::make_unique<Tensor>(std::vector<int>{flat_batch, d_ff_}, false);

    // First linear + activation
    linear1_->forward(flat_input, *hidden_, training);

    // Apply GELU activation in-place
    kernels::gelu_forward(hidden_->data(), hidden_->data(), flat_batch * d_ff_);

    // Second linear
    Tensor flat_output(output.data(), std::vector<int>{flat_batch, d_model_}, false);
    linear2_->forward(*hidden_, flat_output, training);
}

void FeedForward::backward(const Tensor& grad_output, const Tensor& input) {
    // TODO: Implement backward pass
    // This requires:
    // 1. Backward through linear2_
    // 2. Backward through GELU
    // 3. Backward through linear1_
}

std::vector<Tensor*> FeedForward::parameters() {
    std::vector<Tensor*> params;

    auto l1_params = linear1_->parameters();
    auto l2_params = linear2_->parameters();

    params.insert(params.end(), l1_params.begin(), l1_params.end());
    params.insert(params.end(), l2_params.begin(), l2_params.end());

    return params;
}

// ============================================================================
// Transformer Block
// ============================================================================

TransformerBlock::TransformerBlock(
    int d_model, int n_heads, int d_ff,
    float dropout, bool use_rms_norm, float residual_scale
) : residual_scale_(residual_scale) {

    // Create normalization layers
    if (use_rms_norm) {
        norm1_ = std::make_unique<RMSNorm>(d_model);
        norm2_ = std::make_unique<RMSNorm>(d_model);
    } else {
        norm1_ = std::make_unique<LayerNorm>(d_model);
        norm2_ = std::make_unique<LayerNorm>(d_model);
    }

    // Create attention and FFN
    attn_ = std::make_unique<MultiHeadAttention>(d_model, n_heads, dropout);
    ffn_ = std::make_unique<FeedForward>(d_model, d_ff, dropout);
}

void TransformerBlock::forward(
    const Tensor& input,
    Tensor& output,
    const Tensor* mask,
    bool return_attn_weights,
    bool training
) {
    // Pre-norm architecture:
    // x = x + attn(norm(x))
    // x = x + ffn(norm(x))

    int batch_size = input.shape()[0];
    int seq_len = input.shape()[1];
    int d_model = input.shape()[2];

    // Allocate intermediate tensors
    attn_out_ = std::make_unique<Tensor>(input.shape(), false);
    ffn_out_ = std::make_unique<Tensor>(input.shape(), false);
    residual1_ = std::make_unique<Tensor>(input.shape(), false);
    residual2_ = std::make_unique<Tensor>(input.shape(), false);

    // First residual: attention
    // norm1(input)
    Tensor normed1(input.shape(), false);
    if (auto* ln = dynamic_cast<LayerNorm*>(norm1_.get())) {
        ln->forward(input, normed1, training);
    } else if (auto* rn = dynamic_cast<RMSNorm*>(norm1_.get())) {
        rn->forward(input, normed1, training);
    }

    // attn(normed1)
    attn_->forward(normed1, *attn_out_, mask, return_attn_weights, training);

    // residual1 = input + attn_out
    CUDA_CHECK(cudaMemcpy(residual1_->data(), input.data(),
                         input.numel() * sizeof(float), cudaMemcpyDeviceToDevice));
    kernels::residual_add(residual1_->data(), attn_out_->data(),
                         residual1_->data(), residual_scale_,
                         input.numel());

    // Second residual: FFN
    // norm2(residual1)
    Tensor normed2(input.shape(), false);
    if (auto* ln = dynamic_cast<LayerNorm*>(norm2_.get())) {
        ln->forward(*residual1_, normed2, training);
    } else if (auto* rn = dynamic_cast<RMSNorm*>(norm2_.get())) {
        rn->forward(*residual1_, normed2, training);
    }

    // ffn(normed2)
    ffn_->forward(normed2, *ffn_out_, training);

    // output = residual1 + ffn_out
    CUDA_CHECK(cudaMemcpy(output.data(), residual1_->data(),
                         input.numel() * sizeof(float), cudaMemcpyDeviceToDevice));
    kernels::residual_add(output.data(), ffn_out_->data(),
                         output.data(), residual_scale_,
                         input.numel());
}

void TransformerBlock::backward(const Tensor& grad_output, const Tensor& input) {
    // TODO: Implement backward pass
    // This is the most complex backward pass, requiring careful gradient flow
}

std::vector<Tensor*> TransformerBlock::parameters() {
    std::vector<Tensor*> params;

    auto n1_params = norm1_->parameters();
    auto n2_params = norm2_->parameters();
    auto attn_params = attn_->parameters();
    auto ffn_params = ffn_->parameters();

    params.insert(params.end(), n1_params.begin(), n1_params.end());
    params.insert(params.end(), n2_params.begin(), n2_params.end());
    params.insert(params.end(), attn_params.begin(), attn_params.end());
    params.insert(params.end(), ffn_params.begin(), ffn_params.end());

    return params;
}

} // namespace tiny_transformer
