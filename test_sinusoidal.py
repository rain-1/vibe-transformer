"""
Quick test to verify sinusoidal position encoding works.
"""

import torch
from train import TrainingConfig, train


def test_sinusoidal_positions():
    """Test transformer with sinusoidal position encodings."""
    print("=" * 80)
    print("Testing Sinusoidal Position Encodings")
    print("=" * 80)

    config = TrainingConfig(
        vocab_size=16,
        d_model=32,
        n_heads=4,
        d_ff=64,
        max_seq_len=4,
        num_epochs=5,  # Short test
        learning_rate=1e-3,
        batch_size=32,
        train_size=200,
        val_size=50,
        use_sinusoidal_pos=True,  # Enable sinusoidal positions
        device="cpu"
    )

    print("\nConfiguration:")
    print(f"  Position encoding: Sinusoidal (fixed, no learned params)")
    print(f"  d_model: {config.d_model}")
    print(f"  max_seq_len: {config.max_seq_len}")
    print()

    train(config, task="copy")

    print("\n" + "=" * 80)
    print("Sinusoidal position encoding test completed!")
    print("=" * 80)


if __name__ == "__main__":
    test_sinusoidal_positions()
