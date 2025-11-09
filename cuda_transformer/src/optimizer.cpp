/**
 * Optimizer implementations
 */

#include "../include/optimizer.h"
#include "../include/kernels.h"
#include <iostream>

namespace tiny_transformer {

// ============================================================================
// AdamW Optimizer
// ============================================================================

AdamW::AdamW(const std::vector<Tensor*>& parameters, const Config& config)
    : parameters_(parameters), config_(config), step_(0) {
    // Allocate moment buffers for each parameter
    for (auto* param : parameters_) {
        if (param->requires_grad()) {
            m_.push_back(zeros(param->shape(), false));
            v_.push_back(zeros(param->shape(), false));
        }
    }

    std::cout << "AdamW optimizer initialized with " << parameters_.size()
              << " parameter tensors, lr=" << config_.lr << std::endl;
}

void AdamW::step() {
    step_++;

    int param_idx = 0;
    for (size_t i = 0; i < parameters_.size(); i++) {
        auto* param = parameters_[i];

        if (!param->requires_grad() || param->grad() == nullptr) {
            continue;
        }

        // Get moment buffers
        auto& m = m_[param_idx];
        auto& v = v_[param_idx];

        // Call AdamW kernel
        kernels::adamw_update(
            param->data(),
            param->grad(),
            m->data(),
            v->data(),
            config_.lr,
            config_.beta1,
            config_.beta2,
            config_.eps,
            config_.weight_decay,
            step_,
            param->numel()
        );

        param_idx++;
    }
}

void AdamW::zero_grad() {
    for (auto* param : parameters_) {
        if (param->requires_grad()) {
            param->zero_grad();
        }
    }
}

} // namespace tiny_transformer
