"""Generate figures for the Picoder paper from a run's training log.

Reads checkpoints/<run>/train_log.jsonl and writes publication-ready figures
(PNG for previews, PDF for LaTeX) into docs/figures/. Re-run this after any
training run to refresh the figures; it is read-only with respect to the run.

Usage:
    python make_figures.py
    python make_figures.py --out-dir checkpoints/pico --fig-dir docs/figures
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")  # headless backend, no display needed
import matplotlib.pyplot as plt


def load_log(out_dir: str):
    """Parse train_log.jsonl into per-step train events and eval events."""
    path = os.path.join(out_dir, "train_log.jsonl")
    train_steps, train_losses = [], []
    eval_steps, eval_train, eval_val, eval_lr, eval_time = [], [], [], [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ev = rec.get("event")
            if ev == "train":
                train_steps.append(rec["step"])
                train_losses.append(rec["loss"])
            elif ev == "eval":
                eval_steps.append(rec["step"])
                eval_train.append(rec["train_loss"])
                eval_val.append(rec["val_loss"])
                eval_lr.append(rec["lr"])
                eval_time.append(rec["elapsed_s"])
    return {
        "train_steps": train_steps, "train_losses": train_losses,
        "eval_steps": eval_steps, "eval_train": eval_train,
        "eval_val": eval_val, "eval_lr": eval_lr, "eval_time": eval_time,
    }


def save(fig, fig_dir: str, name: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(fig_dir, f"{name}.{ext}"),
                    dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {name}.png and {name}.pdf")


def fig_loss_curves(d, fig_dir: str) -> None:
    """Figure 1: training and validation loss vs step."""
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    # Faint raw per-step train loss (noisy) in the background.
    ax.plot(d["train_steps"], d["train_losses"], color="tab:blue", alpha=0.25,
            linewidth=1.0, label="train (raw, every 50 steps)")
    # Bold matched eval points for train and val.
    ax.plot(d["eval_steps"], d["eval_train"], color="tab:blue", marker="o",
            markersize=3, linewidth=1.5, label="train (eval)")
    ax.plot(d["eval_steps"], d["eval_val"], color="tab:red", marker="s",
            markersize=3, linewidth=1.5, label="validation")
    best = min(d["eval_val"])
    best_step = d["eval_steps"][d["eval_val"].index(best)]
    ax.annotate(f"best val {best:.4f}", xy=(best_step, best),
                xytext=(best_step - 1500, best + 0.35),
                arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
                fontsize=9)
    ax.set_xlabel("training step")
    ax.set_ylabel("cross-entropy loss (nats)")
    ax.set_title("Picoder 0.1: training and validation loss")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    save(fig, fig_dir, "fig1_loss_curves")


def fig_lr_schedule(d, fig_dir: str) -> None:
    """Figure 2: learning-rate schedule (warmup + cosine decay)."""
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    ax.plot(d["eval_steps"], d["eval_lr"], color="tab:green", marker="o",
            markersize=3, linewidth=1.5)
    ax.set_xlabel("training step")
    ax.set_ylabel("learning rate")
    ax.set_title("Picoder 0.1: learning-rate schedule (warmup + cosine decay)")
    ax.grid(True, alpha=0.3)
    save(fig, fig_dir, "fig2_lr_schedule")


def fig_generalization_gap(d, fig_dir: str) -> None:
    """Figure 3: generalization gap (val - train) vs step."""
    gap = [v - t for v, t in zip(d["eval_val"], d["eval_train"])]
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    ax.plot(d["eval_steps"], gap, color="tab:purple", marker="o",
            markersize=3, linewidth=1.5)
    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("training step")
    ax.set_ylabel("val loss - train loss (nats)")
    ax.set_title("Picoder 0.1: generalization gap")
    ax.grid(True, alpha=0.3)
    save(fig, fig_dir, "fig3_generalization_gap")


def fig_loss_vs_time(d, fig_dir: str) -> None:
    """Figure 4: validation loss vs wall-clock time (CPU)."""
    minutes = [t / 60.0 for t in d["eval_time"]]
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    ax.plot(minutes, d["eval_val"], color="tab:red", marker="s",
            markersize=3, linewidth=1.5)
    ax.set_xlabel("wall-clock time (minutes, CPU)")
    ax.set_ylabel("validation loss (nats)")
    ax.set_title("Picoder 0.1: validation loss vs wall-clock time")
    ax.grid(True, alpha=0.3)
    save(fig, fig_dir, "fig4_loss_vs_time")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Picoder paper figures.")
    parser.add_argument("--out-dir", default="checkpoints/pico")
    parser.add_argument("--fig-dir", default="docs/figures")
    args = parser.parse_args()

    os.makedirs(args.fig_dir, exist_ok=True)
    d = load_log(args.out_dir)
    if not d["eval_steps"]:
        raise SystemExit(f"No eval events found in {args.out_dir}/train_log.jsonl")

    fig_loss_curves(d, args.fig_dir)
    fig_lr_schedule(d, args.fig_dir)
    fig_generalization_gap(d, args.fig_dir)
    fig_loss_vs_time(d, args.fig_dir)
    print(f"\nFigures written to {args.fig_dir}/ (PNG + PDF).")
    print(f"Final val loss: {min(d['eval_val']):.4f}")


if __name__ == "__main__":
    main()
