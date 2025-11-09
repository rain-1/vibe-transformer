"""
Tiny Transformer Reference Implementation
A minimal but complete transformer block with numerical stability features.
Designed for educational purposes and as a reference for CUDA implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Tuple


class StableSoftmax(nn.Module):
    """Numerically stable softmax using log-sum-exp trick."""

    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Subtract max for numerical stability
        x_max = x.max(dim=self.dim, keepdim=True)[0]
        x_shifted = x - x_max
        exp_x = torch.exp(x_shifted)
        return exp_x / exp_x.sum(dim=self.dim, keepdim=True)


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    More stable than LayerNorm and faster (no mean computation).
    """

    def __init__(self, d_model: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMS normalization: x / sqrt(mean(x^2) + eps)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        normalized = x / rms
        return self.weight * normalized


class StableLayerNorm(nn.Module):
    """
    Standard LayerNorm with safe epsilon handling.
    """

    def __init__(self, d_model: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        # Add eps inside sqrt for better numerical stability
        normalized = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight * normalized + self.bias


class SinusoidalPositionEncoding(nn.Module):
    """
    Sinusoidal position encoding from 'Attention is All You Need'.

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Benefits:
    - No learned parameters (reduces model size)
    - Can extrapolate to longer sequences than seen in training
    - Encodes relative positions naturally
    """

    def __init__(self, d_model: int, max_seq_len: int = 5000):
        super().__init__()
        self.d_model = d_model

        # Create position encoding matrix
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)

        # Compute the div term: 10000^(2i/d_model) = exp(2i * -log(10000) / d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Apply sin to even indices
        pe[:, 0::2] = torch.sin(position * div_term)

        # Apply cos to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a parameter, but part of state)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)

        Returns:
            Position encodings of shape (batch, seq_len, d_model)
        """
        seq_len = x.size(1)
        return self.pe[:seq_len, :].unsqueeze(0).expand(x.size(0), -1, -1)


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention with numerical stability features.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        use_stable_softmax: bool = True
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # QKV projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Output projection
        self.W_o = nn.Linear(d_model, d_model)

        # Use stable softmax by default
        if use_stable_softmax:
            self.softmax = StableSoftmax(dim=-1)
        else:
            self.softmax = lambda x: F.softmax(x, dim=-1)

        self.dropout = nn.Dropout(dropout)

        # Initialize with variance scaling
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using variance scaling (Xavier/Glorot)."""
        std = 1.0 / math.sqrt(self.d_model)
        for module in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attn_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            mask: Optional attention mask
            return_attn_weights: Whether to return attention weights

        Returns:
            output: Attention output of shape (batch, seq_len, d_model)
            attn_weights: Optional attention weights if return_attn_weights=True
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        Q = self.W_q(x)  # (batch, seq_len, d_model)
        K = self.W_k(x)
        V = self.W_v(x)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        # Now: (batch, n_heads, seq_len, d_k)

        # Scaled dot-product attention
        # Scale by 1/sqrt(d_k) for numerical stability
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # scores: (batch, n_heads, seq_len, seq_len)

        # Apply mask if provided (e.g., for causal attention)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Apply stable softmax
        attn_weights = self.softmax(scores)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        # attn_output: (batch, n_heads, seq_len, d_k)

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)

        # Final projection
        output = self.W_o(attn_output)

        if return_attn_weights:
            return output, attn_weights
        return output, None


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

        # Initialize with variance scaling
        self._init_weights(d_model, d_ff)

    def _init_weights(self, d_model: int, d_ff: int):
        """Initialize weights using variance scaling."""
        std1 = 1.0 / math.sqrt(d_model)
        std2 = 1.0 / math.sqrt(d_ff)

        nn.init.normal_(self.linear1.weight, mean=0.0, std=std1)
        nn.init.normal_(self.linear2.weight, mean=0.0, std=std2)
        nn.init.zeros_(self.linear1.bias)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # FFN(x) = max(0, xW1 + b1)W2 + b2
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerBlock(nn.Module):
    """
    Single transformer block with pre-norm architecture.
    Pre-norm is more stable than post-norm for deep transformers.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        use_rms_norm: bool = False,
        residual_scaling: float = 1.0
    ):
        super().__init__()

        # Layer norms
        norm_layer = RMSNorm if use_rms_norm else StableLayerNorm
        self.norm1 = norm_layer(d_model)
        self.norm2 = norm_layer(d_model)

        # Attention and FFN
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)

        self.dropout = nn.Dropout(dropout)
        self.residual_scaling = residual_scaling

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attn_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Pre-norm transformer block:
        x = x + Attention(Norm(x))
        x = x + FFN(Norm(x))
        """
        # Self-attention with residual
        normed = self.norm1(x)
        attn_out, attn_weights = self.attention(normed, mask, return_attn_weights)
        attn_out = self.dropout(attn_out)
        x = x + self.residual_scaling * attn_out

        # Feed-forward with residual
        normed = self.norm2(x)
        ffn_out = self.ffn(normed)
        ffn_out = self.dropout(ffn_out)
        x = x + self.residual_scaling * ffn_out

        return x, attn_weights


class TinyTransformer(nn.Module):
    """
    Tiny transformer model for educational purposes.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        max_seq_len: int,
        n_blocks: int = 1,
        dropout: float = 0.1,
        use_rms_norm: bool = False,
        use_sinusoidal_pos: bool = False
    ):
        super().__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.use_sinusoidal_pos = use_sinusoidal_pos

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Position encoding: sinusoidal (fixed) or learned
        if use_sinusoidal_pos:
            self.position_encoding = SinusoidalPositionEncoding(d_model, max_seq_len)
        else:
            self.position_embedding = nn.Embedding(max_seq_len, d_model)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, use_rms_norm)
            for _ in range(n_blocks)
        ])

        # Final layer norm
        norm_layer = RMSNorm if use_rms_norm else StableLayerNorm
        self.final_norm = norm_layer(d_model)

        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)

        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize all weights with proper scaling."""
        # Token embeddings
        std = 1.0 / math.sqrt(self.d_model)
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=std)

        # Position embeddings (only if using learned positions)
        if not self.use_sinusoidal_pos:
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=std)

        # Output projection
        nn.init.normal_(self.output_projection.weight, mean=0.0, std=std)
        if self.output_projection.bias is not None:
            nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attn_weights: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Input token indices of shape (batch, seq_len)
            mask: Optional attention mask
            return_attn_weights: Whether to return attention weights

        Returns:
            Dictionary containing:
                - logits: Output logits of shape (batch, seq_len, vocab_size)
                - attn_weights: Optional attention weights from each block
        """
        batch_size, seq_len = x.shape

        # Embed tokens
        token_emb = self.token_embedding(x)  # (batch, seq_len, d_model)

        # Add position encoding
        if self.use_sinusoidal_pos:
            pos_emb = self.position_encoding(token_emb)
        else:
            positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
            pos_emb = self.position_embedding(positions)

        # Combine embeddings
        h = token_emb + pos_emb
        h = self.dropout(h)

        # Apply transformer blocks
        attn_weights_list = []
        for block in self.blocks:
            h, attn_weights = block(h, mask, return_attn_weights)
            if return_attn_weights:
                attn_weights_list.append(attn_weights)

        # Final normalization and projection
        h = self.final_norm(h)
        logits = self.output_projection(h)

        result = {"logits": logits}
        if return_attn_weights:
            result["attn_weights"] = attn_weights_list

        return result

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Create a causal (autoregressive) mask for attention.
    Returns a mask where position i can only attend to positions <= i.
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)


if __name__ == "__main__":
    # Quick test
    print("Testing Tiny Transformer...")

    # Tiny configuration
    vocab_size = 16
    d_model = 32
    n_heads = 4
    d_ff = 64
    max_seq_len = 4
    batch_size = 2

    # Create model
    model = TinyTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        n_blocks=1
    )

    print(f"Model parameters: {model.count_parameters():,}")

    # Create dummy input
    x = torch.randint(0, vocab_size, (batch_size, max_seq_len))

    # Forward pass
    output = model(x, return_attn_weights=True)
    logits = output["logits"]

    print(f"Input shape: {x.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Attention weights shape: {output['attn_weights'][0].shape}")

    # Test with causal mask
    mask = create_causal_mask(max_seq_len, x.device)
    output_masked = model(x, mask=mask)

    print("\nTest passed!")
