"""
Compare learned vs sinusoidal position encodings.
Run both to completion to see which learns better.
"""

import torch
from train import TrainingConfig, train


def test_learned_positions():
    """Test with learned position embeddings."""
    print("=" * 80)
    print("TEST 1: LEARNED POSITION EMBEDDINGS")
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
        train_size=1000,
        val_size=200,
        use_sinusoidal_pos=False,  # Learned positions
        log_interval=20,
        eval_interval=100,
        device="cpu"
    )

    print("\nConfiguration:")
    print(f"  Position encoding: LEARNED (128 parameters)")
    print(f"  d_model: {config.d_model}")
    print(f"  max_seq_len: {config.max_seq_len}")
    print()

    train(config, task="copy")


def test_sinusoidal_positions():
    """Test with sinusoidal position encodings."""
    print("\n\n")
    print("=" * 80)
    print("TEST 2: SINUSOIDAL POSITION ENCODINGS")
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
        train_size=1000,
        val_size=200,
        use_sinusoidal_pos=True,  # Sinusoidal positions
        log_interval=20,
        eval_interval=100,
        device="cpu"
    )

    print("\nConfiguration:")
    print(f"  Position encoding: SINUSOIDAL (0 parameters, fixed)")
    print(f"  d_model: {config.d_model}")
    print(f"  max_seq_len: {config.max_seq_len}")
    print()

    train(config, task="copy")


if __name__ == "__main__":
    print("COMPARING POSITION ENCODING STRATEGIES")
    print("=" * 80)
    print()

    # Run both tests
    test_learned_positions()
    test_sinusoidal_positions()

    print("\n\n")
    print("=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)
    print("\nCheck the final validation losses and accuracies above.")
    print("Both should achieve near-perfect performance on the copy task.")
