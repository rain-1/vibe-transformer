/**
 * Tensor wrapper for CUDA memory management and operations.
 * Provides Python-like interface for GPU tensors with automatic gradient tracking.
 */

#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <string>

namespace tiny_transformer {

class Tensor {
public:
    // Constructors
    Tensor(const std::vector<int>& shape, bool requires_grad = false);
    Tensor(float* data, const std::vector<int>& shape, bool requires_grad = false);
    ~Tensor();

    // Disable copy, enable move
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    // Shape and size
    const std::vector<int>& shape() const { return shape_; }
    int ndim() const { return shape_.size(); }
    int size(int dim) const { return shape_[dim]; }
    int numel() const { return numel_; }

    // Data access
    float* data() { return data_; }
    const float* data() const { return data_; }
    float* grad() { return grad_; }
    const float* grad() const { return grad_; }

    // Gradient handling
    bool requires_grad() const { return requires_grad_; }
    void zero_grad();
    void alloc_grad();  // Allocate gradient buffer if needed

    // Utilities
    void fill_(float value);
    void randn_(float mean = 0.0f, float std = 1.0f);
    void uniform_(float a = 0.0f, float b = 1.0f);

    // Copy operations
    void copy_from_host(const float* host_data);
    void copy_to_host(float* host_data) const;
    void copy_from_device(const float* device_data);

    // Validation
    bool has_nan() const;
    bool has_inf() const;
    float norm() const;  // L2 norm
    float mean() const;
    float std() const;

    // Debugging
    std::string info() const;
    void print(const std::string& name = "", int max_items = 10) const;

private:
    float* data_;
    float* grad_;  // Gradient buffer
    std::vector<int> shape_;
    int numel_;
    bool requires_grad_;
    bool owns_memory_;  // Whether we allocated the memory

    void allocate();
    void deallocate();
    int compute_numel() const;
};

// Factory functions
std::unique_ptr<Tensor> zeros(const std::vector<int>& shape, bool requires_grad = false);
std::unique_ptr<Tensor> ones(const std::vector<int>& shape, bool requires_grad = false);
std::unique_ptr<Tensor> randn(const std::vector<int>& shape, float mean = 0.0f, float std = 1.0f, bool requires_grad = false);
std::unique_ptr<Tensor> uniform(const std::vector<int>& shape, float a = 0.0f, float b = 1.0f, bool requires_grad = false);

} // namespace tiny_transformer
