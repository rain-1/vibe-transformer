/**
 * Tensor implementation - CUDA memory management and operations
 */

#include "../include/tensor.h"
#include "../include/kernels.h"
#include <cuda_runtime.h>
#include <curand.h>
#include <iostream>
#include <sstream>
#include <cmath>
#include <numeric>

namespace tiny_transformer {

// ============================================================================
// Helper functions
// ============================================================================

static curandGenerator_t get_curand_generator() {
    static curandGenerator_t gen = nullptr;
    if (gen == nullptr) {
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
    }
    return gen;
}

// ============================================================================
// Constructors and Destructors
// ============================================================================

Tensor::Tensor(const std::vector<int>& shape, bool requires_grad)
    : data_(nullptr), grad_(nullptr), shape_(shape),
      requires_grad_(requires_grad), owns_memory_(true) {
    numel_ = compute_numel();
    allocate();
}

Tensor::Tensor(float* data, const std::vector<int>& shape, bool requires_grad)
    : data_(data), grad_(nullptr), shape_(shape),
      requires_grad_(requires_grad), owns_memory_(false) {
    numel_ = compute_numel();
}

Tensor::~Tensor() {
    deallocate();
}

Tensor::Tensor(Tensor&& other) noexcept
    : data_(other.data_), grad_(other.grad_), shape_(std::move(other.shape_)),
      numel_(other.numel_), requires_grad_(other.requires_grad_),
      owns_memory_(other.owns_memory_) {
    other.data_ = nullptr;
    other.grad_ = nullptr;
    other.owns_memory_ = false;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        deallocate();

        data_ = other.data_;
        grad_ = other.grad_;
        shape_ = std::move(other.shape_);
        numel_ = other.numel_;
        requires_grad_ = other.requires_grad_;
        owns_memory_ = other.owns_memory_;

        other.data_ = nullptr;
        other.grad_ = nullptr;
        other.owns_memory_ = false;
    }
    return *this;
}

// ============================================================================
// Memory Management
// ============================================================================

void Tensor::allocate() {
    if (numel_ == 0) return;

    if (owns_memory_) {
        CUDA_CHECK(cudaMalloc(&data_, numel_ * sizeof(float)));
        CUDA_CHECK(cudaMemset(data_, 0, numel_ * sizeof(float)));
    }

    if (requires_grad_) {
        alloc_grad();
    }
}

void Tensor::deallocate() {
    if (owns_memory_ && data_ != nullptr) {
        cudaFree(data_);
        data_ = nullptr;
    }

    if (grad_ != nullptr) {
        cudaFree(grad_);
        grad_ = nullptr;
    }
}

int Tensor::compute_numel() const {
    if (shape_.empty()) return 0;
    return std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<int>());
}

// ============================================================================
// Gradient Operations
// ============================================================================

void Tensor::alloc_grad() {
    if (grad_ == nullptr && numel_ > 0) {
        CUDA_CHECK(cudaMalloc(&grad_, numel_ * sizeof(float)));
        CUDA_CHECK(cudaMemset(grad_, 0, numel_ * sizeof(float)));
    }
}

void Tensor::zero_grad() {
    if (grad_ != nullptr) {
        CUDA_CHECK(cudaMemset(grad_, 0, numel_ * sizeof(float)));
    }
}

// ============================================================================
// Initialization
// ============================================================================

void Tensor::fill_(float value) {
    std::vector<float> host_data(numel_, value);
    copy_from_host(host_data.data());
}

void Tensor::randn_(float mean, float std) {
    curandGenerator_t gen = get_curand_generator();
    curandGenerateNormal(gen, data_, numel_, mean, std);
}

void Tensor::uniform_(float a, float b) {
    curandGenerator_t gen = get_curand_generator();
    curandGenerateUniform(gen, data_, numel_);

    // Scale from [0,1] to [a,b]
    float scale = b - a;
    kernels::scale(data_, scale, numel_);

    if (a != 0.0f) {
        // Add offset
        std::vector<float> offset(numel_, a);
        float* d_offset;
        CUDA_CHECK(cudaMalloc(&d_offset, numel_ * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_offset, offset.data(), numel_ * sizeof(float),
                             cudaMemcpyHostToDevice));
        kernels::elementwise_add(data_, d_offset, data_, numel_);
        CUDA_CHECK(cudaFree(d_offset));
    }
}

// ============================================================================
// Copy Operations
// ============================================================================

void Tensor::copy_from_host(const float* host_data) {
    CUDA_CHECK(cudaMemcpy(data_, host_data, numel_ * sizeof(float),
                         cudaMemcpyHostToDevice));
}

void Tensor::copy_to_host(float* host_data) const {
    CUDA_CHECK(cudaMemcpy(host_data, data_, numel_ * sizeof(float),
                         cudaMemcpyDeviceToHost));
}

void Tensor::copy_from_device(const float* device_data) {
    CUDA_CHECK(cudaMemcpy(data_, device_data, numel_ * sizeof(float),
                         cudaMemcpyDeviceToDevice));
}

// ============================================================================
// Validation
// ============================================================================

bool Tensor::has_nan() const {
    return kernels::check_finite(data_, numel_) > 0;
}

bool Tensor::has_inf() const {
    return kernels::check_finite(data_, numel_) > 0;
}

float Tensor::norm() const {
    float sum_sq = 0.0f;
    std::vector<float> host_data(numel_);
    copy_to_host(host_data.data());

    for (int i = 0; i < numel_; i++) {
        sum_sq += host_data[i] * host_data[i];
    }

    return sqrtf(sum_sq);
}

float Tensor::mean() const {
    std::vector<float> host_data(numel_);
    copy_to_host(host_data.data());

    float sum = 0.0f;
    for (int i = 0; i < numel_; i++) {
        sum += host_data[i];
    }

    return sum / numel_;
}

float Tensor::std() const {
    std::vector<float> host_data(numel_);
    copy_to_host(host_data.data());

    float mean_val = mean();
    float var = 0.0f;

    for (int i = 0; i < numel_; i++) {
        float diff = host_data[i] - mean_val;
        var += diff * diff;
    }

    return sqrtf(var / numel_);
}

// ============================================================================
// Debugging
// ============================================================================

std::string Tensor::info() const {
    std::ostringstream oss;
    oss << "Tensor(shape=[";
    for (size_t i = 0; i < shape_.size(); i++) {
        oss << shape_[i];
        if (i < shape_.size() - 1) oss << ", ";
    }
    oss << "], numel=" << numel_;
    oss << ", requires_grad=" << (requires_grad_ ? "true" : "false");
    oss << ", has_grad=" << (grad_ != nullptr ? "true" : "false");
    oss << ")";
    return oss.str();
}

void Tensor::print(const std::string& name, int max_items) const {
    std::vector<float> host_data(std::min(numel_, max_items));
    int items_to_print = std::min(numel_, max_items);

    CUDA_CHECK(cudaMemcpy(host_data.data(), data_, items_to_print * sizeof(float),
                         cudaMemcpyDeviceToHost));

    if (!name.empty()) {
        std::cout << name << ": ";
    }

    std::cout << info() << "\n";
    std::cout << "  data: [";
    for (int i = 0; i < items_to_print; i++) {
        std::cout << host_data[i];
        if (i < items_to_print - 1) std::cout << ", ";
    }
    if (numel_ > max_items) {
        std::cout << ", ...";
    }
    std::cout << "]\n";

    // Print gradient if available
    if (grad_ != nullptr) {
        CUDA_CHECK(cudaMemcpy(host_data.data(), grad_, items_to_print * sizeof(float),
                             cudaMemcpyDeviceToHost));
        std::cout << "  grad: [";
        for (int i = 0; i < items_to_print; i++) {
            std::cout << host_data[i];
            if (i < items_to_print - 1) std::cout << ", ";
        }
        if (numel_ > max_items) {
            std::cout << ", ...";
        }
        std::cout << "]\n";
    }
}

// ============================================================================
// Factory Functions
// ============================================================================

std::unique_ptr<Tensor> zeros(const std::vector<int>& shape, bool requires_grad) {
    auto tensor = std::make_unique<Tensor>(shape, requires_grad);
    CUDA_CHECK(cudaMemset(tensor->data(), 0, tensor->numel() * sizeof(float)));
    return tensor;
}

std::unique_ptr<Tensor> ones(const std::vector<int>& shape, bool requires_grad) {
    auto tensor = std::make_unique<Tensor>(shape, requires_grad);
    tensor->fill_(1.0f);
    return tensor;
}

std::unique_ptr<Tensor> randn(const std::vector<int>& shape, float mean, float std, bool requires_grad) {
    auto tensor = std::make_unique<Tensor>(shape, requires_grad);
    tensor->randn_(mean, std);
    return tensor;
}

std::unique_ptr<Tensor> uniform(const std::vector<int>& shape, float a, float b, bool requires_grad) {
    auto tensor = std::make_unique<Tensor>(shape, requires_grad);
    tensor->uniform_(a, b);
    return tensor;
}

} // namespace tiny_transformer
