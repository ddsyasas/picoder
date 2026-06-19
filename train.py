"""Train Picoder 0.1.

Builds the dataset and tokenizer, instantiates the model, and runs an AdamW
training loop with periodic validation, console plus file logging, and
checkpointing. Everything needed to reproduce a run (config, tokenizer, optimizer
state, step) is saved into the output directory.

Usage:
    python train.py --config configs/pico.yaml
    python train.py --config configs/pico.yaml --max-steps 200   # quick smoke test
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Dict

import torch

from src.config import PicoderConfig
from src.data import build_dataset, get_batch
from src.model import Picoder


def resolve_device(requested: str) -> str:
    """Turn 'auto' into the best available device string."""
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_lr(step: int, cfg: PicoderConfig) -> float:
    """Learning-rate schedule: linear warmup then cosine decay to min_lr."""
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / cfg.warmup_steps
    if not cfg.lr_decay:
        return cfg.learning_rate
    # Cosine decay from learning_rate down to min_lr over the remaining steps.
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    progress = min(1.0, progress)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


@torch.no_grad()
def evaluate(model: Picoder, dataset, cfg: PicoderConfig, device: str,
             generator: torch.Generator) -> Dict[str, float]:
    """Average the loss over eval_iters batches for train and val splits."""
    model.eval()
    out: Dict[str, float] = {}
    for split, ids in (("train", dataset.train_ids), ("val", dataset.val_ids)):
        losses = torch.zeros(cfg.eval_iters)
        for i in range(cfg.eval_iters):
            x, y = get_batch(ids, cfg.batch_size, cfg.block_size, device, generator)
            _, loss = model(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def save_checkpoint(path: str, model: Picoder, optimizer, cfg: PicoderConfig,
                    step: int, best_val: float) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg.as_dict(),
            "step": step,
            "best_val_loss": best_val,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Picoder 0.1.")
    parser.add_argument("--config", default="configs/pico.yaml")
    # Optional overrides for quick experiments without editing the YAML.
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    cfg = PicoderConfig.from_yaml(args.config)
    if args.max_steps is not None:
        cfg.max_steps = args.max_steps
    if args.device is not None:
        cfg.device = args.device
    if args.out_dir is not None:
        cfg.out_dir = args.out_dir
    if args.seed is not None:
        cfg.seed = args.seed
    if args.run_name is not None:
        cfg.run_name = args.run_name

    device = resolve_device(cfg.device)
    torch.manual_seed(cfg.seed)
    # A dedicated generator makes batch sampling reproducible and independent of
    # any other RNG use.
    gen = torch.Generator().manual_seed(cfg.seed)

    os.makedirs(cfg.out_dir, exist_ok=True)
    log_path = os.path.join(cfg.out_dir, "train_log.jsonl")
    log_file = open(log_path, "a")

    def log(record: dict) -> None:
        log_file.write(json.dumps(record) + "\n")
        log_file.flush()

    # --- data ---
    dataset = build_dataset(cfg.data_dir, cfg.dataset, cfg.val_fraction)
    cfg.vocab_size = dataset.vocab_size
    dataset.tokenizer.save(os.path.join(cfg.out_dir, "tokenizer.json"))
    cfg.to_yaml(os.path.join(cfg.out_dir, "config.yaml"))

    print(f"device={device} dtype={cfg.dtype}")
    print(f"vocab_size={cfg.vocab_size}")
    print(f"train tokens={len(dataset.train_ids):,}  val tokens={len(dataset.val_ids):,}")

    # --- model ---
    model = Picoder(cfg).to(device)
    n_params = model.num_params()
    print(f"parameters: {n_params:,} ({n_params/1e6:.2f}M)")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight_decay,
    )

    log({"event": "start", "device": device, "vocab_size": cfg.vocab_size,
         "n_params": n_params, "train_tokens": len(dataset.train_ids),
         "val_tokens": len(dataset.val_ids), "config": cfg.as_dict()})

    # --- training loop ---
    model.train()
    best_val = float("inf")
    t0 = time.time()
    for step in range(cfg.max_steps + 1):
        # Set the scheduled learning rate for this step.
        lr = get_lr(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # Periodic evaluation and checkpointing.
        if step % cfg.eval_interval == 0 or step == cfg.max_steps:
            metrics = evaluate(model, dataset, cfg, device, gen)
            elapsed = time.time() - t0
            print(f"step {step:5d} | train {metrics['train']:.4f} | "
                  f"val {metrics['val']:.4f} | lr {lr:.2e} | {elapsed:.1f}s")
            log({"event": "eval", "step": step, "train_loss": metrics["train"],
                 "val_loss": metrics["val"], "lr": lr, "elapsed_s": elapsed})
            if metrics["val"] < best_val:
                best_val = metrics["val"]
                save_checkpoint(os.path.join(cfg.out_dir, "best.pt"),
                                model, optimizer, cfg, step, best_val)
            save_checkpoint(os.path.join(cfg.out_dir, "latest.pt"),
                            model, optimizer, cfg, step, best_val)

        if step == cfg.max_steps:
            break

        # One optimization step.
        x, y = get_batch(dataset.train_ids, cfg.batch_size, cfg.block_size, device, gen)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        if step % cfg.log_interval == 0:
            print(f"  step {step:5d} | loss {loss.item():.4f} | lr {lr:.2e}")
            log({"event": "train", "step": step, "loss": loss.item(), "lr": lr})

    total = time.time() - t0
    print(f"done in {total:.1f}s | best val loss {best_val:.4f}")
    log({"event": "done", "total_s": total, "best_val_loss": best_val})
    log_file.close()


if __name__ == "__main__":
    main()
