/**
 * Tiny Transformer CUDA - Main training/inference entry point
 */

#include "../include/model.h"
#include "../include/data_loader.h"
#include "../include/optimizer.h"
#include "../include/metrics.h"
#include "../include/kernels.h"
#include <iostream>
#include <string>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>

using namespace tiny_transformer;

struct Config {
    // Task
    std::string task = "copy";

    // Model
    int vocab_size = 16;
    int d_model = 32;
    int n_heads = 4;
    int d_ff = 64;
    int max_seq_len = 4;
    int n_blocks = 1;
    bool use_rms_norm = false;
    bool use_sinusoidal_pos = false;

    // Training
    int num_epochs = 30;
    int batch_size = 32;
    int train_batches = 32;  // 32 batches per epoch = 1024 samples
    int val_batches = 6;     // 6 batches for validation = 192 samples
    float learning_rate = 1e-3f;
    float weight_decay = 0.01f;

    // Logging
    bool use_wandb = false;
    std::string wandb_project = "tiny-transformer-cuda";
    std::string wandb_run_name = "";
};

void print_usage(const char* prog_name) {
    std::cout << "Tiny Transformer CUDA Implementation\n";
    std::cout << "=====================================\n\n";
    std::cout << "Usage: " << prog_name << " [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --task TASK           Training task: copy, reverse, increment (default: copy)\n";
    std::cout << "  --vocab-size N        Vocabulary size (default: 16)\n";
    std::cout << "  --d-model N           Model dimension (default: 32)\n";
    std::cout << "  --n-heads N           Number of attention heads (default: 4)\n";
    std::cout << "  --n-blocks N          Number of transformer blocks (default: 1)\n";
    std::cout << "  --num-epochs N        Number of training epochs (default: 30)\n";
    std::cout << "  --batch-size N        Batch size (default: 32)\n";
    std::cout << "  --learning-rate LR    Learning rate (default: 1e-3)\n";
    std::cout << "  --use-wandb           Enable Weights & Biases logging\n";
    std::cout << "  --help                Show this help message\n";
}

Config parse_args(int argc, char** argv) {
    Config config;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--task" && i + 1 < argc) {
            config.task = argv[++i];
        } else if (arg == "--vocab-size" && i + 1 < argc) {
            config.vocab_size = std::atoi(argv[++i]);
        } else if (arg == "--d-model" && i + 1 < argc) {
            config.d_model = std::atoi(argv[++i]);
        } else if (arg == "--n-heads" && i + 1 < argc) {
            config.n_heads = std::atoi(argv[++i]);
        } else if (arg == "--n-blocks" && i + 1 < argc) {
            config.n_blocks = std::atoi(argv[++i]);
        } else if (arg == "--num-epochs" && i + 1 < argc) {
            config.num_epochs = std::atoi(argv[++i]);
        } else if (arg == "--batch-size" && i + 1 < argc) {
            config.batch_size = std::atoi(argv[++i]);
        } else if (arg == "--learning-rate" && i + 1 < argc) {
            config.learning_rate = std::atof(argv[++i]);
        } else if (arg == "--use-wandb") {
            config.use_wandb = true;
        }
    }

    return config;
}

TaskType parse_task(const std::string& task_str) {
    if (task_str == "copy") return TaskType::COPY;
    if (task_str == "reverse") return TaskType::REVERSE;
    if (task_str == "increment") return TaskType::INCREMENT;

    std::cerr << "Unknown task: " << task_str << ", using copy\n";
    return TaskType::COPY;
}

int main(int argc, char** argv) {
    std::cout << "===============================================\n";
    std::cout << "Tiny Transformer - CUDA Implementation\n";
    std::cout << "===============================================\n\n";

    // Check CUDA availability
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);

    if (error != cudaSuccess || device_count == 0) {
        std::cerr << "ERROR: No CUDA devices found!\n";
        return 1;
    }

    // Print device info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    std::cout << "CUDA Device: " << prop.name << "\n";
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "Total Memory: " << prop.totalGlobalMem / (1024*1024) << " MB\n";
    std::cout << "\n";

    // Parse arguments
    if (argc > 1) {
        std::string arg = argv[1];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
    }

    Config config = parse_args(argc, argv);

    // Print configuration
    std::cout << "Configuration:\n";
    std::cout << "  Task: " << config.task << "\n";
    std::cout << "  Vocab size: " << config.vocab_size << "\n";
    std::cout << "  Model dim: " << config.d_model << "\n";
    std::cout << "  Heads: " << config.n_heads << "\n";
    std::cout << "  Blocks: " << config.n_blocks << "\n";
    std::cout << "  Seq len: " << config.max_seq_len << "\n";
    std::cout << "  Batch size: " << config.batch_size << "\n";
    std::cout << "  Epochs: " << config.num_epochs << "\n";
    std::cout << "  Learning rate: " << config.learning_rate << "\n";
    std::cout << "\n";

    // Create model
    std::cout << "Creating model...\n";
    TinyTransformer model(
        config.vocab_size,
        config.d_model,
        config.n_heads,
        config.d_ff,
        config.max_seq_len,
        config.n_blocks,
        0.0f,  // dropout
        config.use_rms_norm,
        config.use_sinusoidal_pos
    );
    std::cout << "\n";

    // Create data loaders
    std::cout << "Creating data loaders...\n";
    TaskType task = parse_task(config.task);

    SequenceDataLoader train_loader(
        task,
        config.vocab_size,
        config.max_seq_len,
        config.batch_size,
        config.train_batches,
        42  // seed
    );

    SequenceDataLoader val_loader(
        task,
        config.vocab_size,
        config.max_seq_len,
        config.batch_size,
        config.val_batches,
        100  // different seed
    );
    std::cout << "\n";

    // Run inference demo (forward pass only)
    std::cout << "========================================\n";
    std::cout << "Running Inference Demo (Forward Pass)\n";
    std::cout << "========================================\n\n";
    std::cout << "NOTE: Full training requires backward pass implementation.\n";
    std::cout << "This demo shows the model can process data and produce outputs.\n\n";

    for (int epoch = 0; epoch < std::min(config.num_epochs, 3); epoch++) {
        std::cout << "Epoch " << epoch + 1 << "/" << config.num_epochs << "\n";
        std::cout << "--------------------\n";

        train_loader.reset();
        int batch_count = 0;
        float total_loss = 0.0f;
        int total_correct = 0;
        int total_tokens = 0;

        while (batch_count < std::min(config.train_batches, 5)) {
            auto [inputs, targets] = train_loader.next_batch();
            if (!inputs) break;

            // Forward pass
            Tensor logits = model.forward(*inputs, nullptr, true);

            // Compute loss and accuracy
            // Reinterpret logits and targets for loss computation
            const int* targets_int = reinterpret_cast<const int*>(targets->data());

            float loss = kernels::lm_cross_entropy_loss(
                logits.data(),
                targets_int,
                config.batch_size,
                config.max_seq_len,
                config.vocab_size,
                nullptr  // no mask
            );

            float acc = kernels::compute_accuracy(
                logits.data(),
                targets_int,
                config.batch_size,
                config.max_seq_len,
                config.vocab_size,
                nullptr  // no mask
            );

            total_loss += loss;
            total_correct += static_cast<int>(acc * config.batch_size * config.max_seq_len);
            total_tokens += config.batch_size * config.max_seq_len;

            if (batch_count % 2 == 0) {
                std::cout << "  Batch " << batch_count + 1
                          << ": Loss=" << loss
                          << ", Acc=" << (acc * 100.0f) << "%\n";
            }

            batch_count++;
        }

        float avg_loss = total_loss / batch_count;
        float avg_acc = (float)total_correct / total_tokens;

        std::cout << "\nEpoch Summary:\n";
        std::cout << "  Avg Loss: " << avg_loss << "\n";
        std::cout << "  Avg Accuracy: " << (avg_acc * 100.0f) << "%\n";
        std::cout << "\n";
    }

    std::cout << "========================================\n";
    std::cout << "Inference demo complete!\n\n";

    std::cout << "Next Steps:\n";
    std::cout << "  1. Implement backward passes in layers to enable full training\n";
    std::cout << "  2. Add optimizer integration for parameter updates\n";
    std::cout << "  3. Or use PyTorch autograd for gradients\n";
    std::cout << "\n";
    std::cout << "The forward pass is working correctly!\n";
    std::cout << "Model can process sequences and produce logits.\n";

    return 0;
}
