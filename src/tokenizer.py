"""Character-level tokenizer for Picoder 0.1.

The vocabulary is the set of unique characters in the training text. Encoding
maps each character to its integer id; decoding is the exact inverse. This is
zero-dependency and fully transparent, which is the point for a first
reproducible study. A subword (BPE) tokenizer is planned for 0.2.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional


@dataclass
class CharTokenizer:
    """Maps characters to ids and back.

    Attributes:
        stoi: character -> id.
        itos: id -> character.
    """

    stoi: dict[str, int]
    itos: dict[int, str]

    @classmethod
    def build(cls, text: str) -> "CharTokenizer":
        """Build a tokenizer from raw text by collecting unique characters.

        Characters are sorted so the vocabulary is deterministic across runs.
        """
        chars = sorted(set(text))
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for ch, i in stoi.items()}
        return cls(stoi=stoi, itos=itos)

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str) -> list[int]:
        """Encode a string to a list of integer ids."""
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        """Decode a list of integer ids back to a string."""
        return "".join(self.itos[i] for i in ids)

    def save(self, path: str) -> None:
        """Save the vocabulary to a JSON file."""
        # itos keys are ints; JSON keys must be strings, so we store stoi only
        # and rebuild itos on load.
        with open(path, "w") as f:
            json.dump({"stoi": self.stoi}, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        """Load a tokenizer previously written by save()."""
        with open(path, "r") as f:
            raw = json.load(f)
        stoi = {k: int(v) for k, v in raw["stoi"].items()}
        itos = {i: ch for ch, i in stoi.items()}
        return cls(stoi=stoi, itos=itos)
