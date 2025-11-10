/**
 * Metrics tracking and logging similar to PyTorch reference implementation.
 * Includes WandB integration for experiment tracking.
 */

#pragma once

#include "tensor.h"
#include <string>
#include <vector>
#include <map>
#include <memory>

namespace tiny_transformer {

// ============================================================================
// Metrics Tracker
// ============================================================================

class MetricsTracker {
public:
    struct Config {
        bool use_wandb = false;
        std::string wandb_project = "tiny-transformer-cuda";
        std::string wandb_run_name = "";
        int log_interval = 10;
        int eval_interval = 100;
        int histogram_interval = 200;
        bool check_nan_inf = true;
    };

    // Constructor with custom config
    explicit MetricsTracker(const Config& config);

    // Constructor with default config
    MetricsTracker() : MetricsTracker(Config{}) {}

    ~MetricsTracker();

    // Logging
    void log_scalar(const std::string& name, float value, int step);
    void log_scalars(const std::map<std::string, float>& scalars, int step);
    void log_histogram(const std::string& name, const Tensor& tensor, int step);

    // Training metrics
    void log_train_step(
        int epoch, int step, float loss, float grad_norm,
        float lr, float tokens_per_sec
    );

    void log_validation(
        int step, float val_loss, float perplexity,
        const std::vector<std::pair<std::string, float>>& additional_metrics = {}
    );

    // Gradient analysis
    struct GradientStats {
        float global_norm;
        std::map<std::string, float> layer_norms;
        std::map<std::string, float> weight_norms;
        int nan_count;
        int inf_count;
    };

    GradientStats compute_gradient_stats(
        const std::vector<Tensor*>& parameters,
        const std::vector<std::string>& param_names
    );

    void log_gradient_stats(const GradientStats& stats, int step);

    // Activation analysis
    struct ActivationStats {
        std::map<std::string, float> means;
        std::map<std::string, float> stds;
        std::map<std::string, float> maxs;
        std::map<std::string, float> mins;
        int nan_count;
        int inf_count;
    };

    ActivationStats compute_activation_stats(
        const std::map<std::string, const Tensor*>& activations
    );

    void log_activation_stats(const ActivationStats& stats, int step);

    // NaN/Inf detection
    bool check_for_nan_inf(const Tensor& tensor, const std::string& name);

    // Performance tracking
    void start_timer(const std::string& name);
    float stop_timer(const std::string& name);  // Returns elapsed time in ms

    // Summary
    void print_summary(int epoch, int step);

private:
    Config config_;
    void* wandb_run_;  // Opaque pointer to WandB run
    std::map<std::string, float> timers_;
    bool wandb_initialized_;

    void init_wandb();
    void finish_wandb();
};

// ============================================================================
// Performance Timer
// ============================================================================

class Timer {
public:
    Timer();
    void start();
    float stop();  // Returns elapsed time in ms
    float elapsed() const;

private:
    cudaEvent_t start_event_;
    cudaEvent_t stop_event_;
    bool started_;
};

// ============================================================================
// Throughput Calculator
// ============================================================================

class ThroughputCalculator {
public:
    ThroughputCalculator();

    void start_batch(int num_tokens);
    float finish_batch();  // Returns tokens/sec

private:
    Timer timer_;
    int num_tokens_;
};

} // namespace tiny_transformer
