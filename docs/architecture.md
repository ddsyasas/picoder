# Picoder 0.1: Architecture Overview

This is the public, high-level description of the Picoder 0.1 model. It is a
deliberately tiny, fully transparent, decoder-only transformer (GPT style) built
from scratch for research and teaching.

## Model

A stack of identical transformer blocks predicts the next token from the tokens
before it. The pieces, in order:

1. Token embedding: a lookup table of size `vocab_size x n_embd`.
2. Learned positional embedding: `block_size x n_embd`.
3. `n_layer` transformer blocks, each pre-norm:
   - `x = x + Attention(LayerNorm(x))`
   - `x = x + MLP(LayerNorm(x))`
   - Attention is causal multi-head self-attention (each position attends only
     to itself and earlier positions). The MLP is `Linear -> GELU -> Linear`
     with a 4x hidden expansion.
4. A final LayerNorm.
5. A linear output head (weights tied to the token embedding).

The loss is cross-entropy on next-token prediction at every position.

## Default config ("pico")

| Hyperparameter | Value |
| --- | --- |
| vocab_size | 65 (set by the character tokenizer) |
| block_size (context) | 128 |
| n_layer | 4 |
| n_head | 4 |
| head_dim | 32 |
| n_embd | 128 |
| dropout | 0.1 |
| weight tying | yes |
| **parameters** | **818,048 (0.818M)** |

## Tokenizer

Character-level: the vocabulary is the set of unique characters in the training
text. Encoding is per character, decoding is the exact inverse. Zero
dependencies and fully inspectable. A subword (BPE) tokenizer is planned for 0.2.

## Data

TinyShakespeare (about 1 MB of plain text, 1,115,394 characters, 65 distinct
characters), split 90/10 into train and validation by a single contiguous cut.

## Training

AdamW with a short linear warmup followed by cosine learning-rate decay,
gradient clipping, and periodic validation. Every run saves its config and
tokenizer next to the checkpoint so it can be reproduced exactly.

## Generation

Autoregressive sampling with temperature and optional top-k. The context is
cropped to the last `block_size` tokens at each step.

For the full specification and design rationale, see the project repository.
