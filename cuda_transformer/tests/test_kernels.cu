/**
 * Kernel unit tests
 */

#include "../include/kernels.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

using namespace tiny_transformer::kernels;

bool test_softmax() {
    printf("Testing softmax kernel...\n");
    
    // Simple test case
    float input[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float output[4];
    
    float *d_input, *d_output;
    cudaMalloc(&d_input, 4 * sizeof(float));
    cudaMalloc(&d_output, 4 * sizeof(float));
    
    cudaMemcpy(d_input, input, 4 * sizeof(float), cudaMemcpyHostToDevice);
    
    softmax_forward(d_input, d_output, 1, 1, 4);
    
    cudaMemcpy(output, d_output, 4 * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_input);
    cudaFree(d_output);
    
    // Expected: approximately [0.032, 0.087, 0.237, 0.644]
    float expected[4] = {0.0321f, 0.0871f, 0.2369f, 0.6439f};
    
    bool passed = true;
    float max_error = 0.0f;
    for (int i = 0; i < 4; i++) {
        float error = fabsf(output[i] - expected[i]);
        max_error = fmaxf(max_error, error);
        if (error > 1e-3f) {
            passed = false;
        }
    }
    
    printf("  Max error: %.6e\n", max_error);
    printf("  Status: %s\n", passed ? "PASSED" : "FAILED");
    
    return passed;
}

int main() {
    printf("==========================================\n");
    printf("Tiny Transformer - Kernel Unit Tests\n");
    printf("==========================================\n\n");
    
    int passed = 0, total = 0;
    
    // Test softmax
    total++;
    if (test_softmax()) passed++;
    
    printf("\n==========================================\n");
    printf("Results: %d/%d tests passed\n", passed, total);
    printf("==========================================\n");
    
    return (passed == total) ? 0 : 1;
}
