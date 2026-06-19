"""Bigram character-model baseline for Picoder.

Counts character bigrams on the training split and evaluates the per-character
cross-entropy on the validation split, with add-one (Laplace) smoothing. This is
the simplest sensible language model and gives a loss floor the transformers
should beat. It uses the exact same corpus and 90/10 split as the models, so the
numbers are directly comparable.

Reports loss in nats (natural log) and bits-per-character (bpc = nats / ln 2).

Usage:
    python bigram_baseline.py
"""

from __future__ import annotations

import argparse
import math

from src.data import load_corpus
from src.tokenizer import CharTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Bigram baseline for Picoder.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--dataset", default="tinyshakespeare")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    args = parser.parse_args()

    text = load_corpus(args.data_dir, args.dataset)
    tok = CharTokenizer.build(text)
    V = tok.vocab_size

    ids = tok.encode(text)
    n_val = int(len(ids) * args.val_fraction)
    n_train = len(ids) - n_val
    train_ids = ids[:n_train]
    val_ids = ids[n_train:]

    # Count bigrams (prev -> next) on the training split.
    # counts[i][j] = number of times char j follows char i.
    counts = [[0] * V for _ in range(V)]
    for a, b in zip(train_ids[:-1], train_ids[1:]):
        counts[a][b] += 1
    row_tot = [sum(row) for row in counts]

    # Evaluate cross-entropy on the validation split under the add-one smoothed
    # conditional distribution p(next | prev) = (count + 1) / (row_total + V).
    # We sum -log p over each (prev, next) pair in val and average.
    nll = 0.0
    n_pairs = 0
    for a, b in zip(val_ids[:-1], val_ids[1:]):
        p = (counts[a][b] + 1) / (row_tot[a] + V)
        nll += -math.log(p)
        n_pairs += 1

    nats = nll / n_pairs
    bpc = nats / math.log(2)

    print(f"vocab_size: {V}")
    print(f"train chars: {n_train:,}  val chars: {n_val:,}")
    print(f"val pairs scored: {n_pairs:,}")
    print(f"bigram val loss: {nats:.4f} nats  |  {bpc:.4f} bpc")
    print(f"(uniform baseline for reference: {math.log(V):.4f} nats  |  "
          f"{math.log(V)/math.log(2):.4f} bpc)")


if __name__ == "__main__":
    main()
