"""
Test the effect of residual scaling on gradient norms.
Compare 4-block model with and without residual scaling.
"""

from train import TrainingConfig, train


def test_without_residual_scaling():
    """Test 4-block model WITHOUT residual scaling (scale=1.0)."""
    print("=" * 80)
    print("TEST 1: 4-BLOCK MODEL WITHOUT RESIDUAL SCALING (scale=1.0)")
    print("=" * 80)

    config = TrainingConfig(
        vocab_size=32,
        d_model=64,
        n_heads=8,
        d_ff=256,
        max_seq_len=8,
        n_blocks=4,
        num_epochs=10,  # Short test
        learning_rate=5e-4,
        batch_size=32,
        train_size=1000,
        val_size=200,
        residual_scale=1.0,  # NO scaling
        log_interval=20,
        eval_interval=100,
        device="cpu"
    )

    print("\nConfiguration:")
    print(f"  n_blocks: {config.n_blocks}")
    print(f"  Residual scale: {config.residual_scale} (NO SCALING)")
    print(f"  Expected: Higher gradient norms early in training")
    print()

    train(config, task="copy")


def test_with_residual_scaling():
    """Test 4-block model WITH residual scaling (auto: 1/sqrt(8))."""
    print("\n\n")
    print("=" * 80)
    print("TEST 2: 4-BLOCK MODEL WITH RESIDUAL SCALING (auto)")
    print("=" * 80)

    config = TrainingConfig(
        vocab_size=32,
        d_model=64,
        n_heads=8,
        d_ff=256,
        max_seq_len=8,
        n_blocks=4,
        num_epochs=10,  # Short test
        learning_rate=5e-4,
        batch_size=32,
        train_size=1000,
        val_size=200,
        residual_scale=None,  # Auto: 1/sqrt(2*4) = 0.3536
        log_interval=20,
        eval_interval=100,
        device="cpu"
    )

    import math
    expected_scale = 1.0 / math.sqrt(2.0 * config.n_blocks)

    print("\nConfiguration:")
    print(f"  n_blocks: {config.n_blocks}")
    print(f"  Residual scale: Auto = 1/sqrt(2*{config.n_blocks}) = {expected_scale:.4f}")
    print(f"  Expected: Lower gradient norms, more stable training")
    print()

    train(config, task="copy")


if __name__ == "__main__":
    print("TESTING RESIDUAL SCALING EFFECT ON GRADIENT NORMS")
    print("=" * 80)
    print()

    # Run both tests
    test_without_residual_scaling()
    test_with_residual_scaling()

    print("\n\n")
    print("=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)
    print("\nCompare gradient norms in early epochs:")
    print("- WITHOUT scaling (scale=1.0): Higher gradient norms")
    print("- WITH scaling (scale=0.3536): Lower gradient norms")
    print("\nBoth should still converge, but scaled version is more stable.")
