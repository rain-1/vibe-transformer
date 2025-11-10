/**
 * Data loader implementation for sequence tasks
 */

#include "../include/data_loader.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <iostream>

namespace tiny_transformer {

SequenceDataLoader::SequenceDataLoader(
    TaskType task,
    int vocab_size,
    int seq_len,
    int batch_size,
    int num_batches,
    int seed
) : task_(task),
    vocab_size_(vocab_size),
    seq_len_(seq_len),
    batch_size_(batch_size),
    num_batches_(num_batches),
    current_batch_(0),
    rng_(seed) {

    std::cout << "DataLoader initialized: task=";
    switch (task_) {
        case TaskType::COPY: std::cout << "copy"; break;
        case TaskType::REVERSE: std::cout << "reverse"; break;
        case TaskType::INCREMENT: std::cout << "increment"; break;
    }
    std::cout << ", vocab_size=" << vocab_size_
              << ", seq_len=" << seq_len_
              << ", batch_size=" << batch_size_
              << ", num_batches=" << num_batches_
              << std::endl;
}

std::pair<std::unique_ptr<Tensor>, std::unique_ptr<Tensor>>
SequenceDataLoader::next_batch() {
    if (current_batch_ >= num_batches_) {
        return {nullptr, nullptr};  // End of data
    }

    int total_size = batch_size_ * seq_len_;

    // Allocate host memory
    std::vector<int> inputs_host(total_size);
    std::vector<int> targets_host(total_size);

    // Generate data based on task
    switch (task_) {
        case TaskType::COPY:
            generate_copy_batch(inputs_host.data(), targets_host.data());
            break;
        case TaskType::REVERSE:
            generate_reverse_batch(inputs_host.data(), targets_host.data());
            break;
        case TaskType::INCREMENT:
            generate_increment_batch(inputs_host.data(), targets_host.data());
            break;
    }

    // Create tensors and copy to GPU
    // Note: Tensors expect float*, but we store int data as floats
    auto inputs = std::make_unique<Tensor>(
        std::vector<int>{batch_size_, seq_len_}, false
    );
    auto targets = std::make_unique<Tensor>(
        std::vector<int>{batch_size_, seq_len_}, false
    );

    // Copy int data to GPU (cast to float)
    std::vector<float> inputs_float(total_size);
    std::vector<float> targets_float(total_size);
    for (int i = 0; i < total_size; i++) {
        inputs_float[i] = static_cast<float>(inputs_host[i]);
        targets_float[i] = static_cast<float>(targets_host[i]);
    }

    inputs->copy_from_host(inputs_float.data());
    targets->copy_from_host(targets_float.data());

    current_batch_++;
    return {std::move(inputs), std::move(targets)};
}

void SequenceDataLoader::reset() {
    current_batch_ = 0;
}

void SequenceDataLoader::generate_copy_batch(int* inputs, int* targets) {
    // Generate random sequences
    std::uniform_int_distribution<int> dist(0, vocab_size_ - 1);

    for (int i = 0; i < batch_size_ * seq_len_; i++) {
        inputs[i] = dist(rng_);
        targets[i] = inputs[i];  // Copy task: output = input
    }
}

void SequenceDataLoader::generate_reverse_batch(int* inputs, int* targets) {
    // Generate random sequences
    std::uniform_int_distribution<int> dist(0, vocab_size_ - 1);

    for (int b = 0; b < batch_size_; b++) {
        for (int s = 0; s < seq_len_; s++) {
            int idx = b * seq_len_ + s;
            inputs[idx] = dist(rng_);
        }

        // Reverse the sequence for targets
        for (int s = 0; s < seq_len_; s++) {
            int input_idx = b * seq_len_ + s;
            int target_idx = b * seq_len_ + (seq_len_ - 1 - s);
            targets[target_idx] = inputs[input_idx];
        }
    }
}

void SequenceDataLoader::generate_increment_batch(int* inputs, int* targets) {
    // Generate random sequences
    std::uniform_int_distribution<int> dist(0, vocab_size_ - 1);

    for (int i = 0; i < batch_size_ * seq_len_; i++) {
        inputs[i] = dist(rng_);
        targets[i] = (inputs[i] + 1) % vocab_size_;  // Increment with wrap-around
    }
}

} // namespace tiny_transformer
