#!/bin/bash

# Build script for Tiny Transformer CUDA implementation

set -e  # Exit on error

echo "================================"
echo "Tiny Transformer CUDA Build"
echo "================================"
echo

# Configuration
BUILD_TYPE=${1:-Release}  # Release or Debug
BUILD_DIR="build"
INSTALL_PREFIX=${INSTALL_PREFIX:-/usr/local}

# GPU architecture (auto-detect if nvcc available)
if command -v nvcc &> /dev/null; then
    GPU_ARCH=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p')
    echo "Detected CUDA version: $GPU_ARCH"
    # Map CUDA version to architecture
    if [ "${GPU_ARCH%%.*}" -ge "12" ]; then
        CUDA_ARCHS="80;86;89;90"  # Ampere, Ada, Hopper
    elif [ "${GPU_ARCH%%.*}" -ge "11" ]; then
        CUDA_ARCHS="75;80;86"     # Turing, Ampere
    else
        CUDA_ARCHS="75"            # Turing minimum
    fi
else
    echo "WARNING: nvcc not found. Using default architecture."
    CUDA_ARCHS="80;86"
fi

echo "CUDA Architectures: $CUDA_ARCHS"
echo "Build type: $BUILD_TYPE"
echo

# Clean previous build
if [ -d "$BUILD_DIR" ]; then
    echo "Cleaning previous build..."
    rm -rf "$BUILD_DIR"
fi

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure
echo "Configuring with CMake..."
cmake .. \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHS" \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# Build
echo
echo "Building..."
make -j$(nproc)

# Test
echo
echo "Running unit tests..."
if [ -f ./test_kernels ]; then
    ./test_kernels || echo "WARNING: Some tests failed"
else
    echo "WARNING: test_kernels not built"
fi

echo
echo "================================"
echo "Build complete!"
echo "================================"
echo "Executables:"
echo "  - train_transformer: Main training binary"
echo "  - test_kernels: Unit tests"
echo "  - compare_with_pytorch: Validation tool"
echo
echo "To install: sudo make install"
echo "To run: ./train_transformer --help"
