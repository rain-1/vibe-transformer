"""
Training script for Tiny Transformer with comprehensive monitoring and logging.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import time
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import numpy as np

from tiny_transformer import TinyTransformer, create_causal_mask

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not available - logging will be to console only")


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Model config
    vocab_size: int = 16
    d_model: int = 32
    n_heads: int = 4
    d_ff: int = 64
    max_seq_len: int = 4
    n_blocks: int = 1
    dropout: float = 0.0  # Disable for initial debugging
    use_rms_norm: bool = False
    use_sinusoidal_pos: bool = False  # Use sinusoidal position encoding

    # Training config
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    warmup_steps: int = 100

    # Optimizer config (AdamW)
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8  # Increase to 1e-6 for fp16

    # Logging config
    log_interval: int = 10  # Log every N steps
    eval_interval: int = 50  # Evaluate every N steps
    histogram_interval: int = 100  # Log histograms every N steps

    # Data config
    train_size: int = 1000
    val_size: int = 200

    # Stability config
    check_nan_inf: bool = True
    use_amp: bool = False  # Mixed precision training
    loss_scale: float = 1.0  # For manual loss scaling

    # Wandb config
    use_wandb: bool = False
    wandb_project: str = "tiny-transformer"
    wandb_run_name: Optional[str] = None

    # Device
    device: str = "cpu"


# ============================================================================
# Dataset Classes
# ============================================================================

class SequenceCopyDataset(Dataset):
    """
    Simple sequence copy task: model must learn to copy input to output.
    Input: [1, 2, 3, 4] -> Output: [1, 2, 3, 4]
    """

    def __init__(self, vocab_size: int, seq_len: int, num_samples: int, seed: int = 42):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples

        # Generate random sequences
        rng = np.random.RandomState(seed)
        self.data = torch.from_numpy(
            rng.randint(0, vocab_size, size=(num_samples, seq_len))
        ).long()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # For copy task, input and target are the same
        return self.data[idx], self.data[idx]


class SequenceReversalDataset(Dataset):
    """
    Sequence reversal task: model must learn to reverse the input sequence.
    Input: [1, 2, 3, 4] -> Output: [4, 3, 2, 1]
    """

    def __init__(self, vocab_size: int, seq_len: int, num_samples: int, seed: int = 42):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples

        # Generate random sequences
        rng = np.random.RandomState(seed)
        self.inputs = torch.from_numpy(
            rng.randint(0, vocab_size, size=(num_samples, seq_len))
        ).long()
        # Reverse sequences
        self.targets = torch.flip(self.inputs, dims=[1])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


class IncrementDataset(Dataset):
    """
    Simple increment task: each output token is input token + 1 (mod vocab_size).
    Input: [1, 2, 3, 4] -> Output: [2, 3, 4, 5]
    """

    def __init__(self, vocab_size: int, seq_len: int, num_samples: int, seed: int = 42):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples

        # Generate random sequences
        rng = np.random.RandomState(seed)
        self.inputs = torch.from_numpy(
            rng.randint(0, vocab_size, size=(num_samples, seq_len))
        ).long()
        # Increment by 1, wrap around
        self.targets = (self.inputs + 1) % vocab_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


# ============================================================================
# Metrics and Logging
# ============================================================================

class MetricsTracker:
    """
    Comprehensive metrics tracking for transformer training.
    Tracks all the must-have metrics for debugging and monitoring.
    """

    def __init__(self, model: nn.Module, config: TrainingConfig):
        self.model = model
        self.config = config
        self.step = 0
        self.epoch = 0

        # Metrics storage
        self.metrics: Dict[str, List[float]] = defaultdict(list)

        # For throughput calculation
        self.step_start_time = None
        self.tokens_processed = 0

    def check_nan_inf(self, tensor: torch.Tensor, name: str) -> Dict[str, int]:
        """Check for NaN and Inf in tensors."""
        if not self.config.check_nan_inf:
            return {}

        nan_count = torch.isnan(tensor).sum().item()
        inf_count = torch.isinf(tensor).sum().item()

        if nan_count > 0 or inf_count > 0:
            print(f"WARNING: Found NaN/Inf in {name}:")
            print(f"  NaN count: {nan_count}")
            print(f"  Inf count: {inf_count}")

        return {
            f"{name}_nan_count": nan_count,
            f"{name}_inf_count": inf_count
        }

    def compute_grad_norm(self, parameters) -> Tuple[float, Dict[str, float]]:
        """
        Compute global gradient norm and per-layer gradient norms.
        Returns: (global_norm, per_layer_norms)
        """
        total_norm = 0.0
        per_layer_norms = {}

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                per_layer_norms[f"grad_norm/{name}"] = param_norm
                total_norm += param_norm ** 2

        global_norm = math.sqrt(total_norm)
        return global_norm, per_layer_norms

    def compute_weight_norm(self) -> Dict[str, float]:
        """Compute per-layer weight norms."""
        weight_norms = {}

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param_norm = param.data.norm(2).item()
                weight_norms[f"weight_norm/{name}"] = param_norm

        return weight_norms

    def compute_activation_stats(
        self,
        activations: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Compute activation statistics (mean, std, max, min).
        """
        stats = {}

        for name, activation in activations.items():
            # Detach and move to CPU for stats
            act = activation.detach()

            stats[f"activation/{name}_mean"] = act.mean().item()
            stats[f"activation/{name}_std"] = act.std().item()
            stats[f"activation/{name}_max"] = act.max().item()
            stats[f"activation/{name}_min"] = act.min().item()

            # Check for NaN/Inf
            nan_inf_stats = self.check_nan_inf(act, f"activation/{name}")
            stats.update(nan_inf_stats)

        return stats

    def get_histograms(self) -> Dict[str, torch.Tensor]:
        """
        Get parameter and gradient histograms for logging.
        """
        histograms = {}

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Parameter histogram
                histograms[f"param_hist/{name}"] = param.data.cpu()

                # Gradient histogram
                if param.grad is not None:
                    histograms[f"grad_hist/{name}"] = param.grad.data.cpu()

        return histograms

    def log_step(
        self,
        loss: float,
        grad_norm: float,
        learning_rate: float,
        batch_size: int,
        is_train: bool = True
    ):
        """Log metrics for a single step."""
        prefix = "train" if is_train else "val"

        metrics = {
            f"{prefix}/loss": loss,
            f"{prefix}/perplexity": math.exp(min(loss, 20)),  # Clip to avoid overflow
            "train/grad_norm_global": grad_norm if is_train else 0.0,
            "train/learning_rate": learning_rate if is_train else 0.0,
            "step": self.step,
            "epoch": self.epoch
        }

        # Compute throughput
        if self.step_start_time is not None:
            elapsed = time.time() - self.step_start_time
            if elapsed > 0:
                tokens_per_sec = batch_size * self.config.max_seq_len / elapsed
                metrics["train/tokens_per_sec"] = tokens_per_sec
                metrics["train/step_time_ms"] = elapsed * 1000

        return metrics

    def start_step(self):
        """Mark the start of a step for timing."""
        self.step_start_time = time.time()


# ============================================================================
# Training Functions
# ============================================================================

def get_linear_warmup_lr(step: int, config: TrainingConfig) -> float:
    """
    Linear warmup schedule for learning rate.
    """
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps
    return config.learning_rate


def clip_grad_norm(parameters, max_norm: float) -> float:
    """
    Clip gradients by global norm.
    Returns the global gradient norm before clipping.
    """
    # Compute global norm
    total_norm = 0.0
    for param in parameters:
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2

    total_norm = math.sqrt(total_norm)

    # Clip gradients
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for param in parameters:
            if param.grad is not None:
                param.grad.data.mul_(clip_coef)

    return total_norm


def train_step(
    model: nn.Module,
    batch: Tuple[torch.Tensor, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    tracker: MetricsTracker
) -> Dict[str, float]:
    """
    Perform a single training step.
    """
    model.train()
    tracker.start_step()

    inputs, targets = batch
    inputs = inputs.to(config.device)
    targets = targets.to(config.device)

    # Forward pass
    optimizer.zero_grad()
    outputs = model(inputs)
    logits = outputs["logits"]  # (batch, seq_len, vocab_size)

    # Compute loss
    loss = F.cross_entropy(
        logits.view(-1, config.vocab_size),
        targets.view(-1),
        reduction="mean"
    )

    # Backward pass
    loss.backward()

    # Compute gradient norm before clipping
    grad_norm_before = clip_grad_norm(model.parameters(), config.grad_clip_norm)

    # Check for NaN/Inf in gradients
    grad_metrics = {}
    if config.check_nan_inf:
        for name, param in model.named_parameters():
            if param.grad is not None:
                nan_inf_stats = tracker.check_nan_inf(param.grad, f"grad/{name}")
                grad_metrics.update(nan_inf_stats)

    # Optimizer step
    optimizer.step()

    # Update learning rate
    lr = get_linear_warmup_lr(tracker.step, config)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Collect metrics
    metrics = tracker.log_step(
        loss=loss.item(),
        grad_norm=grad_norm_before,
        learning_rate=lr,
        batch_size=inputs.size(0),
        is_train=True
    )

    # Add gradient metrics
    metrics.update(grad_metrics)

    # Per-layer gradient norms
    if tracker.step % config.log_interval == 0:
        _, per_layer_grad_norms = tracker.compute_grad_norm(model.parameters())
        metrics.update(per_layer_grad_norms)

        # Weight norms
        weight_norms = tracker.compute_weight_norm()
        metrics.update(weight_norms)

    tracker.step += 1

    return metrics


@torch.no_grad()
def eval_step(
    model: nn.Module,
    dataloader: DataLoader,
    config: TrainingConfig,
    tracker: MetricsTracker
) -> Dict[str, float]:
    """
    Evaluate the model on validation set.
    """
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(config.device)
        targets = targets.to(config.device)

        # Forward pass
        outputs = model(inputs)
        logits = outputs["logits"]

        # Compute loss
        loss = F.cross_entropy(
            logits.view(-1, config.vocab_size),
            targets.view(-1),
            reduction="sum"
        )

        total_loss += loss.item()
        total_tokens += targets.numel()

    avg_loss = total_loss / total_tokens

    metrics = {
        "val/loss": avg_loss,
        "val/perplexity": math.exp(min(avg_loss, 20)),
        "step": tracker.step,
        "epoch": tracker.epoch
    }

    return metrics


def train(config: TrainingConfig, task: str = "copy"):
    """
    Main training loop with comprehensive logging.

    Args:
        config: Training configuration
        task: Task to train on ("copy", "reverse", "increment")
    """
    print("=" * 80)
    print("Tiny Transformer Training")
    print("=" * 80)
    print(f"Task: {task}")
    print(f"Device: {config.device}")
    print(f"Vocab size: {config.vocab_size}")
    print(f"Sequence length: {config.max_seq_len}")
    print(f"Model dimension: {config.d_model}")
    print(f"Num heads: {config.n_heads}")
    print(f"Num blocks: {config.n_blocks}")
    print()

    # Initialize wandb if requested
    if config.use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config=config.__dict__
        )
        print("Weights & Biases initialized")
    elif config.use_wandb and not WANDB_AVAILABLE:
        print("WARNING: wandb requested but not available")
        config.use_wandb = False

    # Create datasets
    print("Creating datasets...")
    if task == "copy":
        train_dataset = SequenceCopyDataset(
            config.vocab_size, config.max_seq_len, config.train_size, seed=42
        )
        val_dataset = SequenceCopyDataset(
            config.vocab_size, config.max_seq_len, config.val_size, seed=123
        )
    elif task == "reverse":
        train_dataset = SequenceReversalDataset(
            config.vocab_size, config.max_seq_len, config.train_size, seed=42
        )
        val_dataset = SequenceReversalDataset(
            config.vocab_size, config.max_seq_len, config.val_size, seed=123
        )
    elif task == "increment":
        train_dataset = IncrementDataset(
            config.vocab_size, config.max_seq_len, config.train_size, seed=42
        )
        val_dataset = IncrementDataset(
            config.vocab_size, config.max_seq_len, config.val_size, seed=123
        )
    else:
        raise ValueError(f"Unknown task: {task}")

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print()

    # Create model
    print("Creating model...")
    model = TinyTransformer(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        max_seq_len=config.max_seq_len,
        n_blocks=config.n_blocks,
        dropout=config.dropout,
        use_rms_norm=config.use_rms_norm,
        use_sinusoidal_pos=config.use_sinusoidal_pos
    ).to(config.device)

    num_params = model.count_parameters()
    print(f"Model parameters: {num_params:,}")
    print()

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_eps,
        weight_decay=config.weight_decay
    )

    # Create metrics tracker
    tracker = MetricsTracker(model, config)

    # Training loop
    print("Starting training...")
    print()

    best_val_loss = float('inf')
    global_step = 0

    for epoch in range(config.num_epochs):
        tracker.epoch = epoch

        for batch_idx, batch in enumerate(train_loader):
            # Training step
            metrics = train_step(model, batch, optimizer, config, tracker)

            # Log to console
            if global_step % config.log_interval == 0:
                log_str = f"Epoch {epoch:3d} | Step {global_step:5d} | "
                log_str += f"Loss {metrics['train/loss']:.4f} | "
                log_str += f"Grad Norm {metrics['train/grad_norm_global']:.4f} | "
                log_str += f"LR {metrics['train/learning_rate']:.6f}"
                if "train/tokens_per_sec" in metrics:
                    log_str += f" | Tokens/sec {metrics['train/tokens_per_sec']:.1f}"
                print(log_str)

            # Log to wandb
            if config.use_wandb and global_step % config.log_interval == 0:
                wandb.log(metrics, step=global_step)

            # Log histograms
            if config.use_wandb and global_step % config.histogram_interval == 0:
                histograms = tracker.get_histograms()
                hist_dict = {}
                for name, tensor in histograms.items():
                    hist_dict[name] = wandb.Histogram(tensor.numpy())
                wandb.log(hist_dict, step=global_step)

            # Evaluation
            if global_step % config.eval_interval == 0 and global_step > 0:
                val_metrics = eval_step(model, val_loader, config, tracker)
                val_loss = val_metrics["val/loss"]

                log_str = f">>> Validation | Loss {val_loss:.4f} | "
                log_str += f"Perplexity {val_metrics['val/perplexity']:.4f}"
                print(log_str)

                if config.use_wandb:
                    wandb.log(val_metrics, step=global_step)

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    print(f"    New best validation loss: {best_val_loss:.4f}")

            global_step += 1

        print()

    # Final evaluation
    print("=" * 80)
    print("Training complete!")
    print("Final evaluation...")
    final_metrics = eval_step(model, val_loader, config, tracker)
    print(f"Final validation loss: {final_metrics['val/loss']:.4f}")
    print(f"Final perplexity: {final_metrics['val/perplexity']:.4f}")
    print("=" * 80)

    # Test model on a few examples
    print("\nTesting model on sample sequences:")
    model.eval()
    test_samples = 5
    with torch.no_grad():
        for i in range(test_samples):
            inputs, targets = val_dataset[i]
            inputs_batch = inputs.unsqueeze(0).to(config.device)

            outputs = model(inputs_batch)
            logits = outputs["logits"]
            predictions = logits.argmax(dim=-1).squeeze(0)

            print(f"Input:      {inputs.tolist()}")
            print(f"Target:     {targets.tolist()}")
            print(f"Prediction: {predictions.tolist()}")
            correct = (predictions == targets.to(config.device)).all().item()
            print(f"Correct: {correct}")
            print()

    if config.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    # Default configuration for quick testing
    config = TrainingConfig(
        # Model
        vocab_size=16,
        d_model=32,
        n_heads=4,
        d_ff=64,
        max_seq_len=4,
        n_blocks=1,
        dropout=0.0,

        # Training
        batch_size=32,
        num_epochs=50,
        learning_rate=1e-3,
        grad_clip_norm=1.0,

        # Logging
        log_interval=10,
        eval_interval=50,
        histogram_interval=100,

        # Data
        train_size=1000,
        val_size=200,

        # Wandb (disabled by default)
        use_wandb=False,
        wandb_project="tiny-transformer",
        wandb_run_name="copy-task-test",

        device="cpu"
    )

    # Train on copy task
    train(config, task="copy")
