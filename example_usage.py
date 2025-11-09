"""
Example usage scripts for the Tiny Transformer.
Shows different training configurations and tasks.
"""

from train import TrainingConfig, train


def basic_copy_task():
    """Train on simple copy task with default settings."""
    print("=" * 80)
    print("Example 1: Basic Copy Task")
    print("=" * 80)

    config = TrainingConfig(
        vocab_size=16,
        d_model=32,
        n_heads=4,
        d_ff=64,
        max_seq_len=4,
        num_epochs=30,
        learning_rate=1e-3,
        batch_size=32,
        device="cpu"
    )

    train(config, task="copy")


def reversal_task():
    """Train on sequence reversal task."""
    print("=" * 80)
    print("Example 2: Sequence Reversal Task")
    print("=" * 80)

    config = TrainingConfig(
        vocab_size=16,
        d_model=32,
        n_heads=4,
        d_ff=64,
        max_seq_len=4,
        num_epochs=50,  # Harder task, needs more epochs
        learning_rate=1e-3,
        batch_size=32,
        device="cpu"
    )

    train(config, task="reverse")


def increment_task():
    """Train on increment task."""
    print("=" * 80)
    print("Example 3: Increment Task")
    print("=" * 80)

    config = TrainingConfig(
        vocab_size=16,
        d_model=32,
        n_heads=4,
        d_ff=64,
        max_seq_len=4,
        num_epochs=30,
        learning_rate=1e-3,
        batch_size=32,
        device="cpu"
    )

    train(config, task="increment")


def with_rms_norm():
    """Train using RMSNorm instead of LayerNorm."""
    print("=" * 80)
    print("Example 4: Using RMSNorm")
    print("=" * 80)

    config = TrainingConfig(
        vocab_size=16,
        d_model=32,
        n_heads=4,
        d_ff=64,
        max_seq_len=4,
        num_epochs=30,
        learning_rate=1e-3,
        use_rms_norm=True,  # Use RMSNorm
        device="cpu"
    )

    train(config, task="copy")


def larger_model():
    """Train a larger model configuration."""
    print("=" * 80)
    print("Example 5: Larger Model")
    print("=" * 80)

    config = TrainingConfig(
        vocab_size=32,      # Larger vocab
        d_model=64,         # Larger model dimension
        n_heads=8,          # More attention heads
        d_ff=128,           # Larger FFN
        max_seq_len=8,      # Longer sequences
        num_epochs=50,
        learning_rate=5e-4,  # Lower LR for larger model
        batch_size=16,       # Smaller batch due to larger model
        device="cpu"
    )

    train(config, task="copy")


def aggressive_training():
    """
    Example showing more aggressive training with higher learning rate.
    This demonstrates the importance of gradient clipping.
    """
    print("=" * 80)
    print("Example 6: Aggressive Training (with gradient clipping)")
    print("=" * 80)

    config = TrainingConfig(
        vocab_size=16,
        d_model=32,
        n_heads=4,
        d_ff=64,
        max_seq_len=4,
        num_epochs=30,
        learning_rate=5e-3,      # Higher learning rate
        grad_clip_norm=0.5,       # Stronger gradient clipping
        batch_size=32,
        device="cpu"
    )

    train(config, task="copy")


def with_wandb_logging():
    """
    Example with Weights & Biases logging enabled.
    Note: Requires wandb to be installed and configured.
    """
    print("=" * 80)
    print("Example 7: With W&B Logging")
    print("=" * 80)

    config = TrainingConfig(
        vocab_size=16,
        d_model=32,
        n_heads=4,
        d_ff=64,
        max_seq_len=4,
        num_epochs=30,
        learning_rate=1e-3,
        use_wandb=True,
        wandb_project="tiny-transformer",
        wandb_run_name="copy-task-example",
        device="cpu"
    )

    train(config, task="copy")


def verbose_logging():
    """
    Example with very frequent logging for debugging.
    """
    print("=" * 80)
    print("Example 8: Verbose Logging for Debugging")
    print("=" * 80)

    config = TrainingConfig(
        vocab_size=16,
        d_model=32,
        n_heads=4,
        d_ff=64,
        max_seq_len=4,
        num_epochs=10,
        learning_rate=1e-3,
        log_interval=1,         # Log every step
        eval_interval=10,       # Evaluate frequently
        histogram_interval=20,  # Frequent histogram logging
        device="cpu"
    )

    train(config, task="copy")


if __name__ == "__main__":
    import sys

    examples = {
        "1": ("Basic Copy Task", basic_copy_task),
        "2": ("Sequence Reversal", reversal_task),
        "3": ("Increment Task", increment_task),
        "4": ("RMSNorm Example", with_rms_norm),
        "5": ("Larger Model", larger_model),
        "6": ("Aggressive Training", aggressive_training),
        "7": ("With W&B Logging", with_wandb_logging),
        "8": ("Verbose Logging", verbose_logging),
    }

    if len(sys.argv) > 1:
        choice = sys.argv[1]
        if choice in examples:
            name, func = examples[choice]
            func()
        else:
            print(f"Invalid choice: {choice}")
            print(f"Available examples: {', '.join(examples.keys())}")
    else:
        print("=" * 80)
        print("Tiny Transformer - Example Usage")
        print("=" * 80)
        print("\nAvailable examples:")
        for key, (name, _) in examples.items():
            print(f"  {key}: {name}")
        print("\nUsage: python example_usage.py <number>")
        print("Example: python example_usage.py 1")
        print("\nRunning default example (Basic Copy Task)...")
        print()
        basic_copy_task()
