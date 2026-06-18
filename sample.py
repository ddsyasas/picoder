"""Generate text from a trained Picoder checkpoint.

Loads a checkpoint (model weights, config) and the tokenizer saved alongside it,
then autoregressively generates text from a prompt with temperature and optional
top-k sampling.

Usage:
    python sample.py --checkpoint checkpoints/pico/best.pt --prompt "ROMEO:"
    python sample.py --checkpoint checkpoints/pico/best.pt --temperature 0.8 --top-k 40
"""

from __future__ import annotations

import argparse
import os

import torch

from src.config import PicoderConfig
from src.model import Picoder
from src.tokenizer import CharTokenizer
from train import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample from a Picoder checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default=None,
                        help="path to tokenizer.json (defaults to the checkpoint's dir)")
    parser.add_argument("--prompt", default="\n",
                        help="starting text; default is a single newline")
    parser.add_argument("--max-new-tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    torch.manual_seed(args.seed)

    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = PicoderConfig.from_dict(ckpt["config"])

    tok_path = args.tokenizer or os.path.join(
        os.path.dirname(args.checkpoint), "tokenizer.json"
    )
    tokenizer = CharTokenizer.load(tok_path)

    model = Picoder(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Encode the prompt. Any characters not in the vocab would raise a KeyError,
    # which is intentional: we want to know if a prompt is out of distribution.
    start_ids = tokenizer.encode(args.prompt)
    idx = torch.tensor([start_ids], dtype=torch.long, device=device)

    out = model.generate(
        idx,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    text = tokenizer.decode(out[0].tolist())
    print(text)


if __name__ == "__main__":
    main()
