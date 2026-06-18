"""Configuration for Picoder.

A single dataclass holds every hyperparameter so that any run is fully described
by one object. The same object is serialized to YAML next to each checkpoint,
which makes a run reproducible from the saved config alone.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, fields
from typing import Any, Optional

import yaml


@dataclass
class PicoderConfig:
    """All hyperparameters for a Picoder 0.1 run.

    Grouped into model, data, training, and bookkeeping. Defaults match the
    "pico" config described in pdocs/ARCHITECTURE.md.
    """

    # --- model ---
    # vocab_size is set by the tokenizer at build time (None until known).
    vocab_size: Optional[int] = None
    block_size: int = 128          # context length in tokens
    n_layer: int = 4               # number of transformer blocks
    n_head: int = 4                # attention heads per block
    n_embd: int = 128              # embedding / residual stream width
    dropout: float = 0.1
    tie_weights: bool = True       # share token embedding with output head

    # --- data ---
    dataset: str = "tinyshakespeare"
    data_dir: str = "data"
    val_fraction: float = 0.1      # fraction of the corpus held out for validation

    # --- training ---
    batch_size: int = 32
    max_steps: int = 5000
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.99
    grad_clip: float = 1.0
    warmup_steps: int = 100
    lr_decay: bool = True          # cosine decay to min_lr after warmup
    min_lr: float = 3e-5
    eval_interval: int = 250       # steps between validation evaluations
    eval_iters: int = 200          # batches averaged per evaluation
    log_interval: int = 50         # steps between train-loss console logs

    # --- bookkeeping ---
    seed: int = 1337
    device: str = "auto"           # "auto" | "cpu" | "cuda" | "mps"
    dtype: str = "float32"         # "float32" | "bfloat16" | "float16"
    out_dir: str = "checkpoints/pico"
    run_name: str = "pico"

    def to_yaml(self, path: str) -> None:
        """Write this config to a YAML file."""
        with open(path, "w") as f:
            yaml.safe_dump(asdict(self), f, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str) -> "PicoderConfig":
        """Load a config from YAML, ignoring unknown keys gracefully."""
        with open(path, "r") as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PicoderConfig":
        """Build a config from a dict, keeping only known fields."""
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**{k: v for k, v in raw.items() if k in known})

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
