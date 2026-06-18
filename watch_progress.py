"""Live training progress bar for Picoder.

Reads the JSONL log a run writes (checkpoints/<run>/train_log.jsonl) and renders
a self-refreshing progress bar with the current step, latest loss, and an ETA.
This is a read-only viewer; it does not touch the training process.

Usage:
    python watch_progress.py                         # watch checkpoints/pico
    python watch_progress.py --out-dir checkpoints/pico
    python watch_progress.py --once                  # print one line and exit
"""

from __future__ import annotations

import argparse
import json
import os
import time

import yaml


def read_state(out_dir: str):
    """Return (last_step, max_steps, last_loss, last_val, elapsed_s) from the logs."""
    cfg_path = os.path.join(out_dir, "config.yaml")
    log_path = os.path.join(out_dir, "train_log.jsonl")

    max_steps = None
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            max_steps = (yaml.safe_load(f) or {}).get("max_steps")

    last_step = 0
    last_loss = None
    last_val = None
    elapsed_s = None
    done = False
    if os.path.exists(log_path):
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ev = rec.get("event")
                if ev == "train":
                    last_step = max(last_step, rec.get("step", last_step))
                    last_loss = rec.get("loss", last_loss)
                elif ev == "eval":
                    last_step = max(last_step, rec.get("step", last_step))
                    last_val = rec.get("val_loss", last_val)
                    elapsed_s = rec.get("elapsed_s", elapsed_s)
                elif ev == "done":
                    done = True
    return last_step, max_steps, last_loss, last_val, elapsed_s, done


def render(out_dir: str, width: int = 40) -> tuple[str, bool]:
    step, max_steps, loss, val, elapsed, done = read_state(out_dir)
    if not max_steps:
        return ("waiting for the run to start...", False)

    frac = min(1.0, step / max_steps) if max_steps else 0.0
    filled = int(width * frac)
    bar = "#" * filled + "-" * (width - filled)

    # ETA from steps-per-second so far (uses the last eval's elapsed time).
    eta = ""
    if elapsed and step > 0 and not done:
        rate = step / elapsed  # steps per second
        if rate > 0:
            remaining = (max_steps - step) / rate
            m, s = divmod(int(remaining), 60)
            eta = f" | ETA {m:d}m{s:02d}s"

    loss_s = f"{loss:.4f}" if loss is not None else "  -  "
    val_s = f"{val:.4f}" if val is not None else "  -  "
    tag = " DONE" if done else ""
    line = (f"[{bar}] {frac*100:5.1f}%  step {step}/{max_steps}  "
            f"loss {loss_s}  val {val_s}{eta}{tag}")
    return (line, done)


def main() -> None:
    parser = argparse.ArgumentParser(description="Live Picoder training progress.")
    parser.add_argument("--out-dir", default="checkpoints/pico")
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    if args.once:
        line, _ = render(args.out_dir)
        print(line)
        return

    try:
        while True:
            line, done = render(args.out_dir)
            # \r returns to line start; pad to clear any leftover characters.
            print("\r" + line.ljust(100), end="", flush=True)
            if done:
                print()  # newline so the final bar stays on screen
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
