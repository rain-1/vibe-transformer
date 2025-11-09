"""
Debugging utilities for the Tiny Transformer.
Includes finite difference gradient checking and activation inspection.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import numpy as np

from tiny_transformer import TinyTransformer


def finite_difference_grad_check(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-4,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Perform finite difference gradient checking.
    Compares analytical gradients (from backprop) with numerical gradients.

    Args:
        model: The model to check
        inputs: Input tensor
        targets: Target tensor
        eps: Epsilon for finite difference
        verbose: Whether to print detailed results

    Returns:
        Dictionary mapping parameter names to relative errors
    """
    model.eval()
    results = {}

    # Forward pass and backward pass to get analytical gradients
    outputs = model(inputs)
    logits = outputs["logits"]
    loss = nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1)
    )
    loss.backward()

    # Store analytical gradients
    analytical_grads = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            analytical_grads[name] = param.grad.clone()

    # Check each parameter
    for name, param in model.named_parameters():
        if param.grad is None:
            continue

        # Only check a subset of parameters for speed
        param_flat = param.data.view(-1)
        grad_flat = analytical_grads[name].view(-1)

        # Sample a few indices to check (checking all would be too slow)
        num_checks = min(10, len(param_flat))
        indices = np.random.choice(len(param_flat), num_checks, replace=False)

        relative_errors = []

        for idx in indices:
            # Numerical gradient via finite difference
            old_val = param_flat[idx].item()

            # f(x + eps)
            param_flat[idx] = old_val + eps
            outputs_plus = model(inputs)
            loss_plus = nn.functional.cross_entropy(
                outputs_plus["logits"].view(-1, outputs_plus["logits"].size(-1)),
                targets.view(-1)
            )

            # f(x - eps)
            param_flat[idx] = old_val - eps
            outputs_minus = model(inputs)
            loss_minus = nn.functional.cross_entropy(
                outputs_minus["logits"].view(-1, outputs_minus["logits"].size(-1)),
                targets.view(-1)
            )

            # Restore original value
            param_flat[idx] = old_val

            # Numerical gradient
            numerical_grad = (loss_plus.item() - loss_minus.item()) / (2 * eps)
            analytical_grad = grad_flat[idx].item()

            # Relative error
            if abs(numerical_grad) + abs(analytical_grad) > 1e-8:
                rel_error = abs(numerical_grad - analytical_grad) / (
                    abs(numerical_grad) + abs(analytical_grad)
                )
                relative_errors.append(rel_error)

        if relative_errors:
            avg_error = np.mean(relative_errors)
            max_error = np.max(relative_errors)
            results[name] = {"avg_error": avg_error, "max_error": max_error}

            if verbose:
                print(f"{name}:")
                print(f"  Avg relative error: {avg_error:.6e}")
                print(f"  Max relative error: {max_error:.6e}")
                if max_error > 1e-3:
                    print(f"  WARNING: Large gradient error!")

    return results


def inspect_activations(
    model: nn.Module,
    inputs: torch.Tensor,
    verbose: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Inspect intermediate activations in the model.
    Useful for debugging NaN/Inf issues.

    Args:
        model: The model to inspect
        inputs: Input tensor
        verbose: Whether to print statistics

    Returns:
        Dictionary of activation tensors
    """
    model.eval()
    activations = {}

    # Register hooks to capture activations
    hooks = []

    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, dict):
                # Handle model output
                for key, val in output.items():
                    activations[f"{name}_{key}"] = val.detach().clone()
            elif isinstance(output, tuple):
                # Handle multi-output modules
                for i, val in enumerate(output):
                    if val is not None:
                        activations[f"{name}_out{i}"] = val.detach().clone()
            else:
                activations[name] = output.detach().clone()
        return hook

    # Register hooks on all modules
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            hook = module.register_forward_hook(make_hook(name))
            hooks.append(hook)

    # Forward pass
    with torch.no_grad():
        outputs = model(inputs)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Print statistics if verbose
    if verbose:
        print("=" * 80)
        print("Activation Statistics")
        print("=" * 80)

        for name, activation in activations.items():
            if isinstance(activation, torch.Tensor):
                print(f"\n{name}:")
                print(f"  Shape: {activation.shape}")
                print(f"  Mean: {activation.mean().item():.6f}")
                print(f"  Std: {activation.std().item():.6f}")
                print(f"  Min: {activation.min().item():.6f}")
                print(f"  Max: {activation.max().item():.6f}")

                # Check for NaN/Inf
                has_nan = torch.isnan(activation).any().item()
                has_inf = torch.isinf(activation).any().item()

                if has_nan or has_inf:
                    print(f"  WARNING: Contains NaN={has_nan}, Inf={has_inf}")

    return activations


def check_weight_initialization(model: nn.Module, verbose: bool = True):
    """
    Check if weights are properly initialized.

    Args:
        model: The model to check
        verbose: Whether to print details
    """
    if verbose:
        print("=" * 80)
        print("Weight Initialization Check")
        print("=" * 80)

    for name, param in model.named_parameters():
        if param.requires_grad:
            mean = param.data.mean().item()
            std = param.data.std().item()
            abs_max = param.data.abs().max().item()

            if verbose:
                print(f"\n{name}:")
                print(f"  Shape: {param.shape}")
                print(f"  Mean: {mean:.6f}")
                print(f"  Std: {std:.6f}")
                print(f"  Max (abs): {abs_max:.6f}")

            # Check for suspicious initialization
            if abs(mean) > 1.0:
                print(f"  WARNING: Large mean (expected near 0)")
            if std > 1.0:
                print(f"  WARNING: Large std (expected < 1)")
            if abs_max > 10.0:
                print(f"  WARNING: Very large weights")


def test_forward_backward(
    vocab_size: int = 16,
    d_model: int = 32,
    n_heads: int = 4,
    max_seq_len: int = 4,
    batch_size: int = 2
):
    """
    Simple test of forward and backward pass.
    """
    print("=" * 80)
    print("Testing Forward/Backward Pass")
    print("=" * 80)

    # Create model
    model = TinyTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_model * 2,
        max_seq_len=max_seq_len,
        n_blocks=1
    )

    print(f"Model parameters: {model.count_parameters():,}")

    # Create sample input
    inputs = torch.randint(0, vocab_size, (batch_size, max_seq_len))
    targets = torch.randint(0, vocab_size, (batch_size, max_seq_len))

    print(f"\nInput shape: {inputs.shape}")
    print(f"Target shape: {targets.shape}")

    # Check initialization
    check_weight_initialization(model, verbose=False)
    print("\n✓ Weights initialized")

    # Forward pass
    print("\nForward pass...")
    outputs = model(inputs, return_attn_weights=True)
    logits = outputs["logits"]
    print(f"Output shape: {logits.shape}")
    print("✓ Forward pass successful")

    # Check for NaN/Inf in output
    has_nan = torch.isnan(logits).any().item()
    has_inf = torch.isinf(logits).any().item()
    if has_nan or has_inf:
        print(f"✗ WARNING: Output contains NaN={has_nan}, Inf={has_inf}")
    else:
        print("✓ No NaN/Inf in outputs")

    # Compute loss
    print("\nComputing loss...")
    loss = nn.functional.cross_entropy(
        logits.view(-1, vocab_size),
        targets.view(-1)
    )
    print(f"Loss: {loss.item():.4f}")
    print("✓ Loss computation successful")

    # Backward pass
    print("\nBackward pass...")
    loss.backward()
    print("✓ Backward pass successful")

    # Check gradients
    print("\nChecking gradients...")
    grad_count = 0
    nan_grad_count = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_count += 1
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                print(f"✗ WARNING: NaN/Inf in gradient of {name}")
                nan_grad_count += 1

    print(f"✓ Gradients computed for {grad_count} parameters")
    if nan_grad_count == 0:
        print("✓ No NaN/Inf in gradients")
    else:
        print(f"✗ Found NaN/Inf in {nan_grad_count} gradients")

    # Inspect activations
    print("\nInspecting activations...")
    activations = inspect_activations(model, inputs, verbose=False)
    print(f"✓ Captured {len(activations)} activations")

    # Optional: Finite difference check (slow)
    print("\nPerforming gradient check (sampling 10 parameters per layer)...")
    grad_check_results = finite_difference_grad_check(
        model, inputs, targets, verbose=False
    )

    all_good = True
    for name, errors in grad_check_results.items():
        if errors["max_error"] > 1e-3:
            print(f"✗ {name}: max error = {errors['max_error']:.6e}")
            all_good = False

    if all_good:
        print("✓ All gradient checks passed (error < 1e-3)")

    print("\n" + "=" * 80)
    print("All tests passed!" if all_good and nan_grad_count == 0 else "Some issues found")
    print("=" * 80)


if __name__ == "__main__":
    # Run basic tests
    test_forward_backward()
