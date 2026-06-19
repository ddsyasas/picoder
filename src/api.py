"""Convenience API for using a trained Picoder as a library.

This wraps the lower-level model, config, and tokenizer into a one-call loader
and a simple `generate(prompt) -> str` method, so a trained checkpoint can be
used in a few lines:

    from src import load
    model = load("checkpoints/pico/best.pt")
    print(model.generate("ROMEO:", max_new_tokens=200, temperature=0.8))

Picoder is a tiny, character-level research model. It continues text in the
style of its training data; it is not an instruction-following or chat model.
"""

from __future__ import annotations

import os
from typing import Optional

import torch

from .config import PicoderConfig
from .model import Picoder
from .tokenizer import CharTokenizer


def resolve_device(requested: str) -> str:
    """Turn 'auto' into the best available device string."""
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class LoadedPicoder:
    """A trained Picoder ready for text generation.

    Attributes:
        model: the underlying Picoder module (in eval mode).
        tokenizer: the CharTokenizer used at training time.
        config: the PicoderConfig the checkpoint was trained with.
        device: the resolved device string the model lives on.
    """

    def __init__(self, model: Picoder, tokenizer: CharTokenizer,
                 config: PicoderConfig, device: str):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device

    def generate(
        self,
        prompt: str = "\n",
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> str:
        """Generate text that continues `prompt`.

        Args:
            prompt: starting text. Every character must be in the model's
                vocabulary (a KeyError is raised otherwise, on purpose, so an
                out-of-distribution prompt is obvious).
            max_new_tokens: how many characters to generate after the prompt.
            temperature: >1 is more random, <1 is more conservative. Must be > 0.
            top_k: if set, sample only from the k most likely next characters.
            seed: optional RNG seed for reproducible output.

        Returns:
            The prompt followed by the generated continuation, as one string.
        """
        if seed is not None:
            torch.manual_seed(seed)
        start_ids = self.tokenizer.encode(prompt)
        idx = torch.tensor([start_ids], dtype=torch.long, device=self.device)
        out = self.model.generate(
            idx, max_new_tokens=max_new_tokens,
            temperature=temperature, top_k=top_k,
        )
        return self.tokenizer.decode(out[0].tolist())


def load(
    checkpoint: str,
    tokenizer: Optional[str] = None,
    device: str = "auto",
) -> LoadedPicoder:
    """Load a trained Picoder from a checkpoint for generation.

    Args:
        checkpoint: path to a .pt checkpoint saved by train.py.
        tokenizer: path to tokenizer.json. Defaults to `tokenizer.json` in the
            same directory as the checkpoint.
        device: "auto" (default), "cpu", "cuda", or "mps".

    Returns:
        A LoadedPicoder with a `.generate(prompt, ...)` method.
    """
    dev = resolve_device(device)
    ckpt = torch.load(checkpoint, map_location=dev)
    cfg = PicoderConfig.from_dict(ckpt["config"])

    tok_path = tokenizer or os.path.join(
        os.path.dirname(checkpoint), "tokenizer.json"
    )
    tok = CharTokenizer.load(tok_path)

    model = Picoder(cfg).to(dev)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return LoadedPicoder(model=model, tokenizer=tok, config=cfg, device=dev)
