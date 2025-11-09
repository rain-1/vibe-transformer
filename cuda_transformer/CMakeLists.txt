cmake_minimum_required(VERSION 3.18)
project(TinyTransformerCUDA CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)  # Turing, Ampere, Ada

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Optional: cuDNN for optimized operations
find_package(CUDNN)

# Optional: WandB for logging
find_package(PkgConfig)
if(PkgConfig_FOUND)
    pkg_check_modules(WANDB wandb)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math -Xptxas=-v")

# Debug flags
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G -O0")

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Source files
set(CUDA_SOURCES
    kernels/matmul.cu
    kernels/softmax.cu
    kernels/layernorm.cu
    kernels/rmsnorm.cu
    kernels/attention.cu
    kernels/ffn.cu
    kernels/embedding.cu
    kernels/loss.cu
    kernels/optimizer.cu
    kernels/utils.cu
)

set(CPP_SOURCES
    src/tensor.cpp
    src/layers.cpp
    src/model.cpp
    src/optimizer.cpp
    src/trainer.cpp
    src/metrics.cpp
    src/data_loader.cpp
)

# Main library
add_library(tiny_transformer STATIC
    ${CUDA_SOURCES}
    ${CPP_SOURCES}
)

target_link_libraries(tiny_transformer
    CUDA::cublas
    CUDA::cudart
    CUDA::curand
)

if(CUDNN_FOUND)
    target_link_libraries(tiny_transformer ${CUDNN_LIBRARIES})
    target_compile_definitions(tiny_transformer PRIVATE USE_CUDNN)
endif()

if(WANDB_FOUND)
    target_link_libraries(tiny_transformer ${WANDB_LIBRARIES})
    target_compile_definitions(tiny_transformer PRIVATE USE_WANDB)
endif()

# Training executable
add_executable(train_transformer
    src/main.cpp
)

target_link_libraries(train_transformer tiny_transformer)

# Test executable
add_executable(test_kernels
    tests/test_kernels.cu
)

target_link_libraries(test_kernels tiny_transformer)

# Comparison tool (for validating against PyTorch)
add_executable(compare_with_pytorch
    tests/compare_pytorch.cpp
)

target_link_libraries(compare_with_pytorch tiny_transformer)

# Install
install(TARGETS train_transformer test_kernels compare_with_pytorch
    DESTINATION bin
)
