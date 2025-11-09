/**
 * Metrics tracking and logging - Simplified implementation
 */

#include "../include/metrics.h"
#include "../include/kernels.h"
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <chrono>

namespace tiny_transformer {

// ============================================================================
// MetricsTracker
// ============================================================================

MetricsTracker::MetricsTracker(const Config& config)
    : config_(config), wandb_run_(nullptr), wandb_initialized_(false) {
    if (config_.use_wandb) {
        init_wandb();
    }
}

MetricsTracker::~MetricsTracker() {
    if (wandb_initialized_) {
        finish_wandb();
    }
}

void MetricsTracker::init_wandb() {
    // TODO: Implement WandB initialization
    std::cout << "WandB support not yet implemented" << std::endl;
}

void MetricsTracker::finish_wandb() {
    // TODO: Implement WandB finish
}

void MetricsTracker::log_scalar(const std::string& name, float value, int step) {
    std::cout << "Step " << step << " - " << name << ": " << value << std::endl;
}

void MetricsTracker::log_scalars(const std::map<std::string, float>& scalars, int step) {
    for (const auto& [name, value] : scalars) {
        log_scalar(name, value, step);
    }
}

void MetricsTracker::log_histogram(const std::string& name, const Tensor& tensor, int step) {
    // TODO: Implement histogram logging
}

void MetricsTracker::log_train_step(
    int epoch, int step, float loss, float grad_norm,
    float lr, float tokens_per_sec
) {
    std::cout << std::fixed << std::setprecision(4)
              << "Epoch " << epoch
              << " | Step " << step
              << " | Loss: " << loss
              << " | Grad norm: " << grad_norm
              << " | LR: " << lr
              << " | Tokens/sec: " << tokens_per_sec
              << std::endl;
}

void MetricsTracker::log_validation(
    int step, float val_loss, float perplexity,
    const std::vector<std::pair<std::string, float>>& additional_metrics
) {
    std::cout << std::fixed << std::setprecision(4)
              << "Validation @ Step " << step
              << " | Loss: " << val_loss
              << " | Perplexity: " << perplexity
              << std::endl;

    for (const auto& [name, value] : additional_metrics) {
        std::cout << "  " << name << ": " << value << std::endl;
    }
}

MetricsTracker::GradientStats MetricsTracker::compute_gradient_stats(
    const std::vector<Tensor*>& parameters,
    const std::vector<std::string>& param_names
) {
    GradientStats stats;
    stats.global_norm = 0.0f;
    stats.nan_count = 0;
    stats.inf_count = 0;

    // TODO: Implement full gradient statistics computation

    return stats;
}

void MetricsTracker::log_gradient_stats(const GradientStats& stats, int step) {
    std::cout << "Gradient stats @ step " << step
              << " | Global norm: " << stats.global_norm
              << " | NaNs: " << stats.nan_count
              << " | Infs: " << stats.inf_count
              << std::endl;
}

MetricsTracker::ActivationStats MetricsTracker::compute_activation_stats(
    const std::map<std::string, const Tensor*>& activations
) {
    ActivationStats stats;
    stats.nan_count = 0;
    stats.inf_count = 0;

    // TODO: Implement full activation statistics computation

    return stats;
}

void MetricsTracker::log_activation_stats(const ActivationStats& stats, int step) {
    std::cout << "Activation stats @ step " << step << std::endl;
    // TODO: Print detailed stats
}

bool MetricsTracker::check_for_nan_inf(const Tensor& tensor, const std::string& name) {
    if (!config_.check_nan_inf) return false;

    bool has_issues = tensor.has_nan() || tensor.has_inf();
    if (has_issues) {
        std::cerr << "WARNING: NaN/Inf detected in " << name << std::endl;
    }

    return has_issues;
}

void MetricsTracker::start_timer(const std::string& name) {
    timers_[name] = std::chrono::duration<float, std::milli>(
        std::chrono::high_resolution_clock::now().time_since_epoch()
    ).count();
}

float MetricsTracker::stop_timer(const std::string& name) {
    auto now = std::chrono::duration<float, std::milli>(
        std::chrono::high_resolution_clock::now().time_since_epoch()
    ).count();

    float elapsed = now - timers_[name];
    timers_.erase(name);
    return elapsed;
}

void MetricsTracker::print_summary(int epoch, int step) {
    std::cout << "=== Summary: Epoch " << epoch << ", Step " << step << " ===" << std::endl;
}

// ============================================================================
// Timer
// ============================================================================

Timer::Timer() : started_(false) {
    CUDA_CHECK(cudaEventCreate(&start_event_));
    CUDA_CHECK(cudaEventCreate(&stop_event_));
}

void Timer::start() {
    CUDA_CHECK(cudaEventRecord(start_event_));
    started_ = true;
}

float Timer::stop() {
    CUDA_CHECK(cudaEventRecord(stop_event_));
    CUDA_CHECK(cudaEventSynchronize(stop_event_));

    float elapsed_ms;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_event_, stop_event_));

    started_ = false;
    return elapsed_ms;
}

float Timer::elapsed() const {
    if (!started_) return 0.0f;

    cudaEvent_t now;
    CUDA_CHECK(cudaEventCreate(&now));
    CUDA_CHECK(cudaEventRecord(now));
    CUDA_CHECK(cudaEventSynchronize(now));

    float elapsed_ms;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_event_, now));
    CUDA_CHECK(cudaEventDestroy(now));

    return elapsed_ms;
}

// ============================================================================
// ThroughputCalculator
// ============================================================================

ThroughputCalculator::ThroughputCalculator() : num_tokens_(0) {}

void ThroughputCalculator::start_batch(int num_tokens) {
    num_tokens_ = num_tokens;
    timer_.start();
}

float ThroughputCalculator::finish_batch() {
    float elapsed_ms = timer_.stop();
    float tokens_per_sec = (num_tokens_ / elapsed_ms) * 1000.0f;
    return tokens_per_sec;
}

} // namespace tiny_transformer
