/**
 * Data loaders for simple sequence tasks
 */

#pragma once

#include "tensor.h"
#include <memory>
#include <random>

namespace tiny_transformer {

enum class TaskType {
    COPY,      // Output = Input
    REVERSE,   // Output = reversed(Input)
    INCREMENT  // Output = (Input + 1) % vocab_size
};

/**
 * Simple data generator for sequence tasks
 * Generates batches on-the-fly using CPU random generation and CUDA upload
 */
class SequenceDataLoader {
public:
    SequenceDataLoader(
        TaskType task,
        int vocab_size,
        int seq_len,
        int batch_size,
        int num_batches,
        int seed = 42
    );

    /**
     * Generate next batch
     * Returns pair of (inputs, targets) tensors on GPU
     * Both are [batch_size, seq_len] with int values stored as floats
     */
    std::pair<std::unique_ptr<Tensor>, std::unique_ptr<Tensor>> next_batch();

    void reset();  // Reset to beginning

    int num_batches() const { return num_batches_; }
    int current_batch() const { return current_batch_; }

private:
    TaskType task_;
    int vocab_size_;
    int seq_len_;
    int batch_size_;
    int num_batches_;
    int current_batch_;
    std::mt19937 rng_;

    void generate_copy_batch(int* inputs, int* targets);
    void generate_reverse_batch(int* inputs, int* targets);
    void generate_increment_batch(int* inputs, int* targets);
};

} // namespace tiny_transformer
