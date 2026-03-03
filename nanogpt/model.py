"""
NanoGPT - Minimal GPT implementation for educational purposes.
Based on Andrej Karpathy's nanoGPT (https://github.com/karpathy/nanoGPT).

Architecture: GPT-2 style decoder-only transformer.
  - Token + positional embeddings
  - N blocks of: LayerNorm → CausalSelfAttention → LayerNorm → MLP
  - Final LayerNorm → linear projection to vocab size
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


@dataclass
class GPTConfig:
    block_size: int = 256       # maximum sequence length (context window)
    vocab_size: int = 65        # number of tokens (65 for char-level Shakespeare)
    n_layer: int = 6            # number of transformer blocks
    n_head: int = 6             # number of attention heads
    n_embd: int = 384           # embedding dimension
    dropout: float = 0.2        # dropout probability
    bias: bool = True           # use bias in Linear and LayerNorm layers


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal (masked) self-attention.

    Each token attends only to previous tokens (causal mask),
    which is what makes this a language model (not bidirectional).
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout

        # Single projection for Q, K, V (3x for efficiency)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout  = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask: lower-triangular matrix prevents attending to future tokens
        self.register_buffer(
            'bias',
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # batch, sequence length, embedding dim

        # Compute Q, K, V in one shot and split
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # Reshape to (B, n_head, T, head_dim) for multi-head attention
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention: softmax(QK^T / sqrt(d_k)) * V
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale          # (B, H, T, T)
        attn = attn.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = attn @ v                                      # (B, H, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # reassemble heads

        return self.resid_dropout(self.c_proj(out))


class MLP(nn.Module):
    """
    Position-wise feed-forward network (applied identically to each token).
    Expands to 4x the embedding dimension, then projects back.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    """
    One transformer block:
      x = x + Attention(LayerNorm(x))   # residual connection
      x = x + MLP(LayerNorm(x))         # residual connection
    Pre-norm formulation (LayerNorm before sublayer) for stable training.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp  = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """
    GPT Language Model.

    Decoder-only transformer that predicts the next token at every position.
    During training: minimize cross-entropy loss over all positions.
    During inference: sample autoregressively from the predicted distribution.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict({
            'wte':  nn.Embedding(config.vocab_size, config.n_embd),   # token embeddings
            'wpe':  nn.Embedding(config.block_size, config.n_embd),   # position embeddings
            'drop': nn.Dropout(config.dropout),
            'h':    nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            'ln_f': nn.LayerNorm(config.n_embd, bias=config.bias),    # final layer norm
        })
        # Language model head: project embedding → vocab logits
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: share weights between token embedding and output projection
        # (common GPT-2 trick that reduces parameters and improves performance)
        self.transformer['wte'].weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)
        # Scale residual projections by 1/sqrt(2 * n_layer) as in GPT-2
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ):
        """
        Args:
            idx:     (B, T) integer token indices
            targets: (B, T) integer token indices (next-token targets for training)

        Returns:
            logits: (B, T, vocab_size)
            loss:   scalar cross-entropy loss (only if targets provided)
        """
        B, T = idx.shape
        assert T <= self.config.block_size, \
            f"Sequence length {T} exceeds block_size {self.config.block_size}"

        device = idx.device
        pos = torch.arange(0, T, dtype=torch.long, device=device)  # (T,)

        # Forward pass
        tok_emb = self.transformer['wte'](idx)   # (B, T, n_embd)
        pos_emb = self.transformer['wpe'](pos)   # (T, n_embd) — broadcast over batch
        x = self.transformer['drop'](tok_emb + pos_emb)

        for block in self.transformer['h']:
            x = block(x)

        x = self.transformer['ln_f'](x)
        logits = self.lm_head(x)                 # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Flatten to (B*T, vocab_size) for cross-entropy
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> torch.Tensor:
        """
        Autoregressively generate tokens given a context.

        Args:
            idx:            (B, T) context token indices
            max_new_tokens: number of tokens to generate
            temperature:    >1 = more random, <1 = more deterministic
            top_k:          if set, sample from top-k most probable tokens only

        Returns:
            (B, T + max_new_tokens) token indices
        """
        for _ in range(max_new_tokens):
            # Crop context to block_size if needed
            idx_cond = idx if idx.size(1) <= self.config.block_size \
                       else idx[:, -self.config.block_size:]

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # last position, scale by temp

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    def num_parameters(self, exclude_embedding: bool = True) -> int:
        """Count model parameters (optionally exclude embedding table)."""
        n = sum(p.numel() for p in self.parameters())
        if exclude_embedding:
            n -= self.transformer['wpe'].weight.numel()
        return n
