# Tiny Transformer - PyTorch Reference Implementation

A minimal but complete transformer implementation in PyTorch, designed for educational purposes and as a reference for CUDA implementations. This implementation emphasizes numerical stability, comprehensive logging, and debuggability.

## Features

### Architecture
- **Single transformer block** with multi-head self-attention
- **Numerically stable components**:
  - Stable softmax using log-sum-exp trick
  - LayerNorm with safe epsilon handling
  - RMSNorm as alternative (faster, more stable)
  - Proper weight initialization (variance scaling)
- **Pre-norm architecture** for better gradient flow
- **Configurable model size** (default: d_model=32, n_heads=4)

### Training Tasks
Three simple sequence-to-sequence tasks for testing:
1. **Copy Task**: Learn to copy input sequence to output
2. **Reversal Task**: Learn to reverse the input sequence
3. **Increment Task**: Learn to increment each token by 1 (mod vocab_size)

### Numerical Stability Features
- ✅ Numerically stable softmax (subtract max before exp)
- ✅ Safe LayerNorm with proper epsilon handling
- ✅ Gradient clipping by global norm
- ✅ Proper weight initialization (Xavier/Glorot scaling)
- ✅ Linear warmup for learning rate
- ✅ AdamW optimizer with configurable epsilon
- ✅ NaN/Inf detection and logging
- ✅ Gradient and activation monitoring

### Comprehensive Monitoring & Logging

All must-have metrics for debugging transformer training:

**Per-step metrics:**
- Training loss and validation loss
- Global gradient norm (||g||₂)
- Per-layer gradient norms
- Per-layer weight norms
- Learning rate
- Training throughput (tokens/sec)
- Step timing

**Periodic metrics:**
- Parameter histograms (per-layer)
- Gradient histograms (per-layer)
- Activation statistics (mean, std, max, min)
- NaN/Inf counters (triggers warnings if > 0)

**Optional:**
- Weights & Biases integration for experiment tracking
- Attention weight visualization

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

Run the basic test to verify the implementation:

```bash
python tiny_transformer.py
```

Train on the copy task:

```bash
python train.py
```

### Training on Different Tasks

Edit `train.py` to change the task:

```python
# At the bottom of train.py, change the task parameter:
train(config, task="copy")      # Copy task (default)
train(config, task="reverse")   # Reversal task
train(config, task="increment") # Increment task
```

Or create a custom training script:

```python
from train import TrainingConfig, train

config = TrainingConfig(
    vocab_size=16,
    d_model=32,
    n_heads=4,
    max_seq_len=4,
    num_epochs=50,
    learning_rate=1e-3,
)

train(config, task="copy")
```

### Enable Weights & Biases Logging

```python
config = TrainingConfig(
    # ... other config ...
    use_wandb=True,
    wandb_project="tiny-transformer",
    wandb_run_name="my-experiment",
)
```

## Model Architecture

```
TinyTransformer
├── Token Embedding (vocab_size -> d_model)
├── Position Embedding (max_seq_len -> d_model)
├── TransformerBlock x n_blocks
│   ├── LayerNorm/RMSNorm
│   ├── Multi-Head Self-Attention
│   │   ├── Q, K, V Projections
│   │   ├── Scaled Dot-Product Attention
│   │   ├── Stable Softmax
│   │   └── Output Projection
│   ├── Residual Connection
│   ├── LayerNorm/RMSNorm
│   ├── Feed-Forward Network (d_model -> d_ff -> d_model)
│   └── Residual Connection
├── Final LayerNorm/RMSNorm
└── Output Projection (d_model -> vocab_size)
```

## Configuration

### Model Hyperparameters

```python
TrainingConfig(
    # Model architecture
    vocab_size=16,        # Vocabulary size
    d_model=32,           # Model dimension
    n_heads=4,            # Number of attention heads
    d_ff=64,              # Feed-forward hidden dimension
    max_seq_len=4,        # Maximum sequence length
    n_blocks=1,           # Number of transformer blocks
    dropout=0.0,          # Dropout rate
    use_rms_norm=False,   # Use RMSNorm instead of LayerNorm

    # Training
    batch_size=32,
    num_epochs=50,
    learning_rate=1e-3,
    weight_decay=0.01,
    grad_clip_norm=1.0,
    warmup_steps=100,

    # Optimizer (AdamW)
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_eps=1e-8,        # Increase to 1e-6 for fp16
)
```

## Debugging Tips

### If you see NaN/Inf:

1. **Check the logs** - The tracker will automatically detect and report NaN/Inf in:
   - Activations
   - Gradients
   - Parameters

2. **Reduce learning rate** - Try 10x smaller LR:
   ```python
   config.learning_rate = 1e-4
   ```

3. **Increase gradient clipping**:
   ```python
   config.grad_clip_norm = 0.5
   ```

4. **Check initialization** - Model uses proper variance scaling by default

5. **Monitor attention scores** - Enable attention weight logging:
   ```python
   output = model(x, return_attn_weights=True)
   ```

### If training is unstable:

1. **Enable comprehensive logging**:
   ```python
   config.log_interval = 1  # Log every step
   ```

2. **Check gradient norms** - Look for exploding/vanishing gradients in the logs

3. **Try RMSNorm** instead of LayerNorm:
   ```python
   config.use_rms_norm = True
   ```

4. **Disable dropout** initially:
   ```python
   config.dropout = 0.0
   ```

## Example Output

```
================================================================================
Tiny Transformer Training
================================================================================
Task: copy
Device: cpu
Vocab size: 16
Sequence length: 4
Model dimension: 32
Num heads: 4
Num blocks: 1

Model parameters: 18,448

Starting training...

Epoch   0 | Step     0 | Loss 2.7845 | Grad Norm 4.2341 | LR 0.000010 | Tokens/sec 2048.3
Epoch   0 | Step    10 | Loss 2.3214 | Grad Norm 2.1234 | LR 0.000110 | Tokens/sec 2134.5
>>> Validation | Loss 2.1234 | Perplexity 8.3654

Input:      [3, 7, 2, 11]
Target:     [3, 7, 2, 11]
Prediction: [3, 7, 2, 11]
Correct: True
```

## Reference for CUDA Implementation

This implementation is designed to be a reference for CUDA development. Key aspects:

1. **Clear numerical stability patterns** - Every operation shows how to handle numerical issues
2. **Comprehensive logging** - Track every metric you'll need to debug CUDA kernels
3. **Simple architecture** - One block, small dimensions, easy to verify
4. **Multiple test tasks** - Verify correctness on different patterns
5. **Gradient checking ready** - Can add finite difference checks easily

### Recommended workflow for CUDA development:

1. Train reference model on CPU
2. Log all intermediate activations and gradients
3. Implement CUDA kernel
4. Compare CUDA outputs with PyTorch reference
5. Use comprehensive logging to identify discrepancies

## File Structure

```
.
├── tiny_transformer.py    # Model architecture and components
├── train.py              # Training loop and metrics tracking
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## License

MIT License - See LICENSE file for details.