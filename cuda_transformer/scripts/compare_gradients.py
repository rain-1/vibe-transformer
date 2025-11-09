#!/usr/bin/env python3
"""
Gradient validation script - compares CUDA implementation against PyTorch reference.
Uses finite differences to validate backward passes.
"""

import torch
import numpy as np
import subprocess
import json
import sys
from pathlib import Path

# Add parent directory to path to import PyTorch reference
sys.path.append(str(Path(__file__).parent.parent.parent))
from train import TrainingConfig, train
from tiny_transformer import TinyTransformer


def save_checkpoint(model, path):
    """Save PyTorch model checkpoint for CUDA comparison."""
    checkpoint = {
        'state_dict': model.state_dict(),
        'config': {
            'vocab_size': model.token_embedding.weight.shape[0],
            'd_model': model.d_model,
            'n_heads': model.blocks[0].attention.n_heads,
            'd_ff': model.blocks[0].ffn.linear1.out_features,
            'max_seq_len': model.max_seq_len,
            'n_blocks': len(model.blocks),
        }
    }
    torch.save(checkpoint, path)
    print(f"Saved PyTorch checkpoint to {path}")


def compare_activations(cuda_activations, pytorch_activations):
    """Compare activations from CUDA and PyTorch implementations."""
    print("\nComparing Activations:")
    print("=" * 80)

    max_errors = {}
    for name in pytorch_activations.keys():
        if name not in cuda_activations:
            print(f"WARNING: {name} not found in CUDA activations")
            continue

        cuda_vals = np.array(cuda_activations[name])
        pytorch_vals = pytorch_activations[name].detach().cpu().numpy()

        diff = np.abs(cuda_vals - pytorch_vals)
        max_error = np.max(diff)
        rel_error = max_error / (np.abs(pytorch_vals).max() + 1e-8)

        max_errors[name] = max_error

        status = "PASS" if max_error < 1e-5 else "FAIL"
        print(f"[{status}] {name:30s}: max_error = {max_error:.6e}, rel_error = {rel_error:.6e}")

    return max_errors


def compare_gradients(cuda_grads, pytorch_grads):
    """Compare gradients from CUDA and PyTorch implementations."""
    print("\nComparing Gradients:")
    print("=" * 80)

    max_errors = {}
    for name in pytorch_grads.keys():
        if name not in cuda_grads:
            print(f"WARNING: {name} not found in CUDA gradients")
            continue

        cuda_grad = np.array(cuda_grads[name])
        pytorch_grad = pytorch_grads[name].detach().cpu().numpy()

        diff = np.abs(cuda_grad - pytorch_grad)
        max_error = np.max(diff)
        rel_error = max_error / (np.abs(pytorch_grad).max() + 1e-8)

        max_errors[name] = max_error

        status = "PASS" if max_error < 1e-4 else "FAIL"
        print(f"[{status}] {name:30s}: max_error = {max_error:.6e}, rel_error = {rel_error:.6e}")

    return max_errors


def finite_difference_check(model, input_tensor, target_tensor, eps=1e-4):
    """
    Validate gradients using finite differences.
    This is independent of CUDA - just checks PyTorch gradients.
    """
    print("\nFinite Difference Gradient Check:")
    print("=" * 80)

    model.eval()

    # Forward + backward
    output = model(input_tensor)
    logits = output['logits']
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        target_tensor.view(-1)
    )
    loss.backward()

    # Check a few parameters
    params_to_check = [
        ('token_embedding.weight', model.token_embedding.weight),
        ('blocks.0.attention.W_q.weight', model.blocks[0].attention.W_q.weight),
        ('blocks.0.ffn.linear1.weight', model.blocks[0].ffn.linear1.weight),
    ]

    for name, param in params_to_check:
        if param.grad is None:
            continue

        # Sample a few random indices
        flat_param = param.data.view(-1)
        flat_grad = param.grad.view(-1)

        indices = np.random.choice(len(flat_param), min(10, len(flat_param)), replace=False)

        max_error = 0
        for idx in indices:
            # f(x + eps)
            original = flat_param[idx].item()
            flat_param[idx] = original + eps
            output_plus = model(input_tensor)
            loss_plus = torch.nn.functional.cross_entropy(
                output_plus['logits'].view(-1, output_plus['logits'].size(-1)),
                target_tensor.view(-1)
            )

            # f(x - eps)
            flat_param[idx] = original - eps
            output_minus = model(input_tensor)
            loss_minus = torch.nn.functional.cross_entropy(
                output_minus['logits'].view(-1, output_minus['logits'].size(-1)),
                target_tensor.view(-1)
            )

            # Restore
            flat_param[idx] = original

            # Numerical gradient
            numerical_grad = (loss_plus.item() - loss_minus.item()) / (2 * eps)
            analytical_grad = flat_grad[idx].item()

            # Relative error
            if abs(numerical_grad) + abs(analytical_grad) > 1e-8:
                error = abs(numerical_grad - analytical_grad) / (abs(numerical_grad) + abs(analytical_grad))
                max_error = max(max_error, error)

        status = "PASS" if max_error < 1e-3 else "FAIL"
        print(f"[{status}] {name:40s}: max_rel_error = {max_error:.6e}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Compare CUDA implementation with PyTorch reference')
    parser.add_argument('--cuda-binary', type=str, default='build/train_transformer',
                       help='Path to CUDA binary')
    parser.add_argument('--pytorch-script', type=str, default='../train.py',
                       help='Path to PyTorch training script')
    parser.add_argument('--num-checks', type=int, default=10,
                       help='Number of gradient checks to perform')
    parser.add_argument('--checkpoint', type=str, default='checkpoint.pth',
                       help='Path to save/load checkpoint')

    args = parser.parse_args()

    # Create simple test case
    print("Creating PyTorch reference model...")
    model = TinyTransformer(
        vocab_size=16,
        d_model=32,
        n_heads=4,
        d_ff=64,
        max_seq_len=4,
        n_blocks=1
    )

    # Create sample input
    batch_size = 2
    seq_len = 4
    input_tensor = torch.randint(0, 16, (batch_size, seq_len))
    target_tensor = torch.randint(0, 16, (batch_size, seq_len))

    # Run finite difference check on PyTorch model
    finite_difference_check(model, input_tensor, target_tensor)

    # Save checkpoint for CUDA comparison
    save_checkpoint(model, args.checkpoint)

    print("\n" + "=" * 80)
    print("PyTorch validation complete!")
    print("=" * 80)
    print(f"\nCheckpoint saved to: {args.checkpoint}")
    print(f"To compare with CUDA, run:")
    print(f"  {args.cuda_binary} --pytorch-checkpoint {args.checkpoint}")


if __name__ == '__main__':
    main()
