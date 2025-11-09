"""
Scaled-up training script for Tiny Transformer.
Uses 4 transformer blocks, larger vocab, and bigger datasets.
"""

from train import TrainingConfig, train


def train_scaled_copy():
    """
    Scaled-up copy task with 4 transformer blocks.
    More challenging configuration for testing.
    """
    config = TrainingConfig(
        # Scaled model architecture
        vocab_size=64,           # 4x larger vocab
        d_model=64,              # 2x larger model dimension
        n_heads=8,               # 2x more attention heads
        d_ff=256,                # 4x larger feed-forward
        max_seq_len=8,           # 2x longer sequences
        n_blocks=4,              # 4 transformer blocks
        dropout=0.1,             # Add some dropout
        use_rms_norm=False,

        # Training config
        batch_size=64,           # Larger batch size
        num_epochs=100,          # More epochs for harder task
        learning_rate=5e-4,      # Slightly lower LR for stability
        weight_decay=0.01,
        grad_clip_norm=1.0,
        warmup_steps=200,        # More warmup steps

        # Optimizer
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_eps=1e-8,

        # Logging
        log_interval=20,
        eval_interval=100,
        histogram_interval=200,

        # Larger datasets
        train_size=5000,         # 5x more training samples
        val_size=1000,           # 5x more validation samples

        # Stability
        check_nan_inf=True,
        use_amp=False,

        # Wandb
        use_wandb=False,
        wandb_project="tiny-transformer-scaled",
        wandb_run_name="scaled-copy-4blocks",

        device="cpu"
    )

    print("=" * 80)
    print("SCALED TINY TRANSFORMER")
    print("=" * 80)
    print(f"Model size: {config.n_blocks} blocks, d_model={config.d_model}, "
          f"n_heads={config.n_heads}")
    print(f"Vocab size: {config.vocab_size}")
    print(f"Sequence length: {config.max_seq_len}")
    print(f"Training samples: {config.train_size}")
    print(f"Validation samples: {config.val_size}")
    print("=" * 80)
    print()

    train(config, task="copy")


def train_scaled_reverse():
    """
    Scaled-up reversal task - more challenging than copy.
    """
    config = TrainingConfig(
        # Scaled model architecture
        vocab_size=64,
        d_model=64,
        n_heads=8,
        d_ff=256,
        max_seq_len=8,
        n_blocks=4,
        dropout=0.1,
        use_rms_norm=False,

        # Training config
        batch_size=64,
        num_epochs=150,          # Even more epochs for reversal
        learning_rate=5e-4,
        weight_decay=0.01,
        grad_clip_norm=1.0,
        warmup_steps=200,

        # Logging
        log_interval=20,
        eval_interval=100,

        # Larger datasets
        train_size=5000,
        val_size=1000,

        # Stability
        check_nan_inf=True,

        device="cpu"
    )

    print("=" * 80)
    print("SCALED TINY TRANSFORMER - REVERSAL TASK")
    print("=" * 80)
    print(f"Model size: {config.n_blocks} blocks, d_model={config.d_model}")
    print(f"Task: Sequence reversal (harder than copy)")
    print("=" * 80)
    print()

    train(config, task="reverse")


def train_scaled_increment():
    """
    Scaled-up increment task.
    """
    config = TrainingConfig(
        vocab_size=64,
        d_model=64,
        n_heads=8,
        d_ff=256,
        max_seq_len=8,
        n_blocks=4,
        dropout=0.1,

        batch_size=64,
        num_epochs=100,
        learning_rate=5e-4,
        grad_clip_norm=1.0,
        warmup_steps=200,

        log_interval=20,
        eval_interval=100,

        train_size=5000,
        val_size=1000,

        check_nan_inf=True,
        device="cpu"
    )

    print("=" * 80)
    print("SCALED TINY TRANSFORMER - INCREMENT TASK")
    print("=" * 80)
    print(f"Model size: {config.n_blocks} blocks, d_model={config.d_model}")
    print("=" * 80)
    print()

    train(config, task="increment")


def train_mega_scaled():
    """
    Even larger configuration for serious testing.
    This might be slow on CPU but shows the architecture scales.
    """
    config = TrainingConfig(
        # Large model
        vocab_size=128,          # Even larger vocab
        d_model=128,             # Larger hidden size
        n_heads=8,
        d_ff=512,                # Large FFN
        max_seq_len=16,          # Longer sequences
        n_blocks=6,              # 6 transformer blocks
        dropout=0.1,
        use_rms_norm=True,       # Try RMSNorm for this one

        # Training
        batch_size=32,           # Smaller batch due to model size
        num_epochs=50,           # Fewer epochs due to time
        learning_rate=3e-4,
        weight_decay=0.01,
        grad_clip_norm=1.0,
        warmup_steps=300,

        # Logging
        log_interval=20,
        eval_interval=100,

        # Datasets
        train_size=10000,
        val_size=2000,

        check_nan_inf=True,
        device="cpu"
    )

    print("=" * 80)
    print("MEGA SCALED TINY TRANSFORMER")
    print("=" * 80)
    print(f"Model: {config.n_blocks} blocks, d_model={config.d_model}, "
          f"n_heads={config.n_heads}")
    print(f"Vocab: {config.vocab_size}, Seq len: {config.max_seq_len}")
    print(f"Training samples: {config.train_size}")
    print("WARNING: This will be slow on CPU!")
    print("=" * 80)
    print()

    train(config, task="copy")


def compare_model_sizes():
    """
    Compare training dynamics across different model sizes.
    Quick test to see how model size affects convergence.
    """
    print("=" * 80)
    print("MODEL SIZE COMPARISON")
    print("=" * 80)
    print()

    configs = [
        ("Tiny (1 block)", TrainingConfig(
            vocab_size=32, d_model=32, n_heads=4, d_ff=128,
            max_seq_len=8, n_blocks=1, batch_size=64,
            num_epochs=30, train_size=2000, val_size=400,
            log_interval=20, eval_interval=100
        )),
        ("Small (2 blocks)", TrainingConfig(
            vocab_size=32, d_model=32, n_heads=4, d_ff=128,
            max_seq_len=8, n_blocks=2, batch_size=64,
            num_epochs=30, train_size=2000, val_size=400,
            log_interval=20, eval_interval=100
        )),
        ("Medium (4 blocks)", TrainingConfig(
            vocab_size=32, d_model=64, n_heads=8, d_ff=256,
            max_seq_len=8, n_blocks=4, batch_size=64,
            num_epochs=30, train_size=2000, val_size=400,
            log_interval=20, eval_interval=100
        )),
    ]

    results = []

    for name, config in configs:
        print(f"\n{'='*80}")
        print(f"Training: {name}")
        print(f"{'='*80}\n")

        # Import here to get fresh model each time
        from train import TinyTransformer
        model = TinyTransformer(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            max_seq_len=config.max_seq_len,
            n_blocks=config.n_blocks,
            dropout=config.dropout
        )

        print(f"Parameters: {model.count_parameters():,}")
        print()

        # Train and capture final loss
        train(config, task="copy")

        print(f"\nCompleted: {name}")
        print("=" * 80)
        print()


if __name__ == "__main__":
    import sys

    options = {
        "1": ("Scaled Copy (4 blocks)", train_scaled_copy),
        "2": ("Scaled Reversal (4 blocks)", train_scaled_reverse),
        "3": ("Scaled Increment (4 blocks)", train_scaled_increment),
        "4": ("Mega Scaled (6 blocks)", train_mega_scaled),
        "5": ("Compare Model Sizes", compare_model_sizes),
    }

    if len(sys.argv) > 1:
        choice = sys.argv[1]
        if choice in options:
            name, func = options[choice]
            print(f"\nRunning: {name}\n")
            func()
        else:
            print(f"Invalid choice: {choice}")
            print(f"Available options: {', '.join(options.keys())}")
    else:
        print("=" * 80)
        print("Scaled Tiny Transformer Training")
        print("=" * 80)
        print("\nAvailable configurations:")
        for key, (name, _) in options.items():
            print(f"  {key}: {name}")
        print("\nUsage: python train_scaled.py <number>")
        print("Example: python train_scaled.py 1")
        print("\nRunning default (Scaled Copy Task with 4 blocks)...")
        print()
        train_scaled_copy()
