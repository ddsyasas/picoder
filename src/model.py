"""Picoder 0.1 model: a decoder-only transformer (GPT style).

A stack of identical blocks predicts the next token from the preceding tokens.
The code favors clarity over speed: every tensor shape is annotated so a reader
can follow the data through embeddings, attention, the MLP, and the loss.

Shapes use these letters:
    B = batch size
    T = sequence length (<= block_size)
    C = embedding width (n_embd)
    H = number of heads (n_head)
    V = vocabulary size (vocab_size)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F

from .config import PicoderConfig


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with a causal mask.

    Each position attends only to itself and earlier positions, which is what
    makes this a left-to-right language model.
    """

    def __init__(self, cfg: PicoderConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head

        # One linear produces queries, keys, and values together (3 * C out).
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)  # mixes heads back together
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

        # Lower-triangular mask of shape (1, 1, block_size, block_size). Stored
        # as a buffer so it moves with the model but is not a parameter.
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size))
        self.register_buffer("causal_mask", mask.view(1, 1, cfg.block_size, cfg.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # Project to q, k, v then split. Each is (B, T, C).
        q, k, v = self.qkv(x).split(self.n_embd, dim=2)

        # Reshape to (B, H, T, head_dim) so attention runs per head in parallel.
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention scores: (B, H, T, T).
        # Scaling by 1/sqrt(head_dim) keeps the softmax in a sensible range.
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

        # Causal mask: positions in the future are set to -inf before softmax,
        # so they contribute zero weight. We slice the buffer to the actual T.
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # Weighted sum of values, then merge heads back to (B, T, C).
        y = att @ v                                  # (B, H, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.proj(y))
        return y


class MLP(nn.Module):
    """Position-wise feed-forward network with a 4x hidden expansion."""

    def __init__(self, cfg: PicoderConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.gelu = nn.GELU()
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(self.gelu(self.fc(x))))


class Block(nn.Module):
    """One transformer block: pre-norm attention and MLP, each with a residual.

    Pre-norm (LayerNorm before the sublayer) is the common stable choice and is
    what we use here. The residual additions let gradients flow through the stack.
    """

    def __init__(self, cfg: PicoderConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class Picoder(nn.Module):
    """The full decoder-only transformer."""

    def __init__(self, cfg: PicoderConfig):
        super().__init__()
        assert cfg.vocab_size is not None, "vocab_size must be set before building the model"
        self.cfg = cfg

        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        if cfg.tie_weights:
            # Share weights between the input embedding and the output head.
            # This saves parameters and is standard practice for small LMs.
            self.head.weight = self.token_emb.weight

        self.apply(self._init_weights)
        # Scaled init on residual projections (GPT-2 style) for stable depth.
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = False) -> int:
        """Count parameters. With non_embedding=True, exclude the position table
        (the token table is counted once; it is tied to the head)."""
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.pos_emb.weight.numel()
        return n

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run a forward pass.

        Args:
            idx: input token ids, shape (B, T), with T <= block_size.
            targets: optional next-token ids, shape (B, T). If given, the loss
                is returned alongside the logits.

        Returns:
            logits of shape (B, T, V), and the cross-entropy loss (or None).
        """
        B, T = idx.shape
        assert T <= self.cfg.block_size, (
            f"sequence length {T} exceeds block_size {self.cfg.block_size}"
        )

        pos = torch.arange(T, device=idx.device)            # (T,)
        tok = self.token_emb(idx)                            # (B, T, C)
        x = self.drop(tok + self.pos_emb(pos))               # broadcast pos over batch
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)                                # (B, T, V)

        loss = None
        if targets is not None:
            # Cross-entropy over the next token at every position. Flatten the
            # batch and time dims so each of B*T predictions is one example.
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """Autoregressively extend idx by max_new_tokens.

        Args:
            idx: starting context, shape (B, T).
            temperature: >1 is more random, <1 is more greedy. Must be > 0.
            top_k: if set, sample only from the k most likely next tokens.
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Crop context to the last block_size tokens (the model's window).
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            # Take the last position's logits and apply temperature.
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None:
                k = min(top_k, logits.size(-1))
                v, _ = torch.topk(logits, k)
                # Mask everything below the k-th best to -inf.
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx
