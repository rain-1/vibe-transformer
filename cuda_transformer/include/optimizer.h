/**
 * Optimizer implementations (AdamW, etc.)
 */

#pragma once

#include "tensor.h"
#include <vector>
#include <memory>

namespace tiny_transformer {

class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual void step() = 0;
    virtual void zero_grad() = 0;
    virtual float get_lr() const = 0;
    virtual void set_lr(float lr) = 0;
};

// ============================================================================
// AdamW Optimizer
// ============================================================================

class AdamW : public Optimizer {
public:
    struct Config {
        float lr = 1e-3f;
        float beta1 = 0.9f;
        float beta2 = 0.999f;
        float eps = 1e-8f;
        float weight_decay = 0.01f;
    };

    AdamW(const std::vector<Tensor*>& parameters, const Config& config = {});

    void step() override;
    void zero_grad() override;
    float get_lr() const override { return config_.lr; }
    void set_lr(float lr) override { config_.lr = lr; }

    int get_step() const { return step_; }

private:
    std::vector<Tensor*> parameters_;
    std::vector<std::unique_ptr<Tensor>> m_;  // First moment
    std::vector<std::unique_ptr<Tensor>> v_;  // Second moment
    Config config_;
    int step_;
};

// ============================================================================
// Learning Rate Scheduler
// ============================================================================

class LRScheduler {
public:
    virtual ~LRScheduler() = default;
    virtual float get_lr(int step) const = 0;
};

class LinearWarmup : public LRScheduler {
public:
    LinearWarmup(float base_lr, int warmup_steps)
        : base_lr_(base_lr), warmup_steps_(warmup_steps) {}

    float get_lr(int step) const override {
        if (step < warmup_steps_) {
            return base_lr_ * (step + 1) / warmup_steps_;
        }
        return base_lr_;
    }

private:
    float base_lr_;
    int warmup_steps_;
};

} // namespace tiny_transformer
