"""A compact nanoGPT-style decoder-only Transformer.

This file intentionally keeps the architecture explicit. The goal is not to
hide complexity behind a framework, but to expose the exact components that show
up in GPT-like LLMs: token embeddings, positional embeddings, causal attention,
MLP blocks, residual connections, normalization, and next-token loss.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    """Hyperparameters that define the model size and context length."""

    vocab_size: int
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with a causal mask.

    "Causal" means token position i may only attend to positions <= i. This is
    the key restriction that makes a decoder-only Transformer usable for
    autoregressive text generation: during training, we can process all positions
    in parallel while preventing information leakage from the future.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head

        # One projection produces q, k, and v together for efficiency.
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # The lower-triangular mask is stored as a buffer: it is part of the
        # module state, moves with .to(device), but is not a trainable parameter.
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("causal_mask", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, embd = x.shape

        q, k, v = self.c_attn(x).split(embd, dim=2)

        # Shape convention:
        # (B, T, C) -> (B, T, n_head, head_dim) -> (B, n_head, T, head_dim)
        q = q.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        # Attention scores are scaled dot products between queries and keys.
        # Dividing by sqrt(head_dim) keeps logits numerically well-behaved.
        scores = (q @ k.transpose(-2, -1)) * (self.head_dim**-0.5)
        scores = scores.masked_fill(self.causal_mask[:, :, :seq_len, :seq_len] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)

        # Weighted sum of values, then merge all heads back into one embedding.
        out = weights @ v
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, embd)
        out = self.resid_dropout(self.c_proj(out))
        return out


class MLP(nn.Module):
    """The feed-forward part of a Transformer block."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """A pre-norm Transformer block.

    The residual pattern is:
        x = x + attention(layer_norm(x))
        x = x + mlp(layer_norm(x))

    Pre-norm is widely used because it makes optimization more stable than the
    older post-norm layout when models get deeper.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class MiniGPT(nn.Module):
    """A small GPT that predicts the next token for every input position."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: input token embeddings and output classifier weights
        # share parameters. This reduces parameter count and usually improves LM
        # quality for small decoder-only models.
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights with the GPT-style small normal distribution."""

        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run the model.

        Args:
            idx: token ids with shape (B, T)
            targets: optional next-token ids with shape (B, T)

        Returns:
            logits: unnormalized probabilities with shape (B, T, vocab_size)
            loss: cross-entropy loss when targets are provided, else None
        """

        batch, seq_len = idx.shape
        if seq_len > self.config.block_size:
            raise ValueError(f"sequence length {seq_len} exceeds block_size {self.config.block_size}")

        positions = torch.arange(0, seq_len, dtype=torch.long, device=idx.device)
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(positions)
        x = self.dropout(tok_emb + pos_emb)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # Cross entropy expects shape (N, C), so flatten B*T positions into N.
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Autoregressively append tokens to an existing prompt."""

        self.eval()
        for _ in range(max_new_tokens):
            # The model only supports block_size context tokens. For long prompts,
            # keep the most recent context, which is also how GPT-style inference
            # windows are commonly handled.
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)

            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = -float("inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)

        return idx

    def num_parameters(self) -> int:
        """Return the number of trainable parameters."""

        return sum(p.numel() for p in self.parameters() if p.requires_grad)

