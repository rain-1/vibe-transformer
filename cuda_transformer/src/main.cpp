/**
 * Tiny Transformer CUDA - Main training entry point
 */

#include <iostream>
#include <string>
#include <cuda_runtime.h>

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
    std::cout << "  --residual-scale S    Residual scaling factor (default: auto)\n";
    std::cout << "  --use-wandb           Enable Weights & Biases logging\n";
    std::cout << "  --wandb-project NAME  W&B project name\n";
    std::cout << "  --wandb-run-name NAME W&B run name\n";
    std::cout << "  --help                Show this help message\n";
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
    bool show_help = false;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            show_help = true;
            break;
        }
    }

    if (show_help || argc == 1) {
        print_usage(argv[0]);
        return 0;
    }

    std::cout << "TODO: Implementation in progress\n";
    std::cout << "\n";
    std::cout << "Status:\n";
    std::cout << "  ✓ Build system configured\n";
    std::cout << "  ✓ CUDA device detected\n";
    std::cout << "  ✓ Softmax kernel implemented (see kernels/softmax.cu)\n";
    std::cout << "  ⧗ Remaining kernels to be implemented\n";
    std::cout << "\n";
    std::cout << "Next steps:\n";
    std::cout << "  1. Implement remaining kernels (follow softmax.cu pattern)\n";
    std::cout << "  2. Implement Tensor class\n";
    std::cout << "  3. Implement Layer classes\n";
    std::cout << "  4. Implement training loop\n";
    std::cout << "\n";
    std::cout << "See IMPLEMENTATION_GUIDE.md for details.\n";

    return 0;
}
