"""Data pipeline for Picoder 0.1.

Loads a plain-text corpus, tokenizes it once with a character-level tokenizer,
splits into train and validation, and serves random contiguous windows as
batches. Targets are the inputs shifted by one position (next-token prediction).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import torch

from .tokenizer import CharTokenizer


@dataclass
class Dataset:
    """A tokenized corpus split into train and validation id tensors."""

    tokenizer: CharTokenizer
    train_ids: torch.Tensor   # 1-D long tensor of token ids
    val_ids: torch.Tensor     # 1-D long tensor of token ids

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size


def load_corpus(data_dir: str, dataset: str) -> str:
    """Read the raw corpus text from data_dir/<dataset>.txt."""
    path = os.path.join(data_dir, f"{dataset}.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Corpus not found at {path}. Run `python prepare_data.py` first."
        )
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_dataset(
    data_dir: str,
    dataset: str,
    val_fraction: float,
    tokenizer: CharTokenizer | None = None,
) -> Dataset:
    """Load the corpus, build (or reuse) a tokenizer, encode, and split.

    The split is a hard cut at (1 - val_fraction) of the token stream, not a
    shuffle, so train and validation come from different regions of the text.
    """
    text = load_corpus(data_dir, dataset)
    if tokenizer is None:
        tokenizer = CharTokenizer.build(text)

    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n_val = int(len(ids) * val_fraction)
    n_train = len(ids) - n_val
    train_ids = ids[:n_train]
    val_ids = ids[n_train:]
    return Dataset(tokenizer=tokenizer, train_ids=train_ids, val_ids=val_ids)


def get_batch(
    ids: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: str,
    generator: torch.Generator | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample one batch of (x, y) windows from a 1-D id tensor.

    For each of batch_size sequences we pick a random start index i and take
    x = ids[i : i+block_size] and y = ids[i+1 : i+1+block_size]. So y is x
    shifted one step right: the model predicts the next token at every position.

    Returns:
        x, y each of shape (batch_size, block_size), on the target device.
    """
    # Highest valid start index so that i + block_size + 1 stays in range.
    high = len(ids) - block_size - 1
    ix = torch.randint(0, high, (batch_size,), generator=generator)
    x = torch.stack([ids[i : i + block_size] for i in ix])
    y = torch.stack([ids[i + 1 : i + 1 + block_size] for i in ix])

    if device.startswith("cuda"):
        # Pinned, async transfer is a small win on GPU; harmless to skip on CPU.
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
    return x, y
