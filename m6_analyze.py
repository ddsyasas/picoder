"""Aggregate the M6 scaling sweep into the tables used in the write-up.

Reads every run under checkpoints/m6/<config>_s<seed>/, computes the best and
final validation loss per config as mean +/- std over the seeds (population std,
since the seeds are the whole sample), in both nats and bits-per-character
(bpc = nats / ln 2), and identifies the best seed per config (lowest best val
loss) for downstream word-validity measurement.

Usage:
    python m6_analyze.py
    python m6_analyze.py --m6-dir checkpoints/m6 --json out.json
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os

LN2 = math.log(2.0)

# Display order and friendly labels for the configs.
ORDER = ["m6_tiny", "m6_small", "m6_pico", "m6_big"]


def read_run(run_dir: str):
    """Return (best_val, final_val, best_ckpt) for one run, or None."""
    log_path = os.path.join(run_dir, "train_log.jsonl")
    if not os.path.exists(log_path):
        return None
    vals = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("event") == "eval":
                vals.append(r["val_loss"])
    if not vals:
        return None
    return min(vals), vals[-1], os.path.join(run_dir, "best.pt")


def param_count(cfg_path: str) -> int:
    import torch  # noqa: F401
    from src.config import PicoderConfig
    from src.model import Picoder
    return Picoder(PicoderConfig.from_yaml(cfg_path)).num_params()


def mean_std(xs):
    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / len(xs)
    return m, var ** 0.5


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate the M6 sweep.")
    parser.add_argument("--m6-dir", default="checkpoints/m6")
    parser.add_argument("--json", default=None, help="optional path to dump summary JSON")
    args = parser.parse_args()

    # Group runs by config.
    by_config: dict[str, list[tuple[int, str]]] = {}
    for run_dir in glob.glob(os.path.join(args.m6_dir, "*_s*")):
        base = os.path.basename(run_dir)
        config, seed = base.rsplit("_s", 1)
        by_config.setdefault(config, []).append((int(seed), run_dir))

    configs = [c for c in ORDER if c in by_config] + \
              [c for c in sorted(by_config) if c not in ORDER]

    summary = []
    for config in configs:
        seeds = sorted(by_config[config])
        bests, finals, best_seed, best_seed_val, best_ckpt = [], [], None, math.inf, None
        cfg_path = None
        for seed, run_dir in seeds:
            res = read_run(run_dir)
            if res is None:
                continue
            best, final, ckpt = res
            bests.append(best)
            finals.append(final)
            cfg_path = os.path.join(run_dir, "config.yaml")
            if best < best_seed_val:
                best_seed_val, best_seed, best_ckpt = best, seed, ckpt
        if not bests:
            continue
        bm, bs = mean_std(bests)
        fm, fs = mean_std(finals)
        summary.append({
            "config": config,
            "params": param_count(cfg_path) if cfg_path else None,
            "n_seeds": len(bests),
            "best_nats_mean": bm, "best_nats_std": bs,
            "best_bpc_mean": bm / LN2, "best_bpc_std": bs / LN2,
            "final_nats_mean": fm, "final_nats_std": fs,
            "final_bpc_mean": fm / LN2, "final_bpc_std": fs / LN2,
            "best_seed": best_seed, "best_seed_val": best_seed_val,
            "best_seed_ckpt": best_ckpt,
        })

    summary.sort(key=lambda r: (r["params"] or 0))

    # Markdown table: best val loss in nats and bpc, mean +/- std.
    print("\n| config | params | best val (nats) | best val (bpc) | final val (nats) | final val (bpc) |")
    print("| --- | ---: | --- | --- | --- | --- |")
    for r in summary:
        print(f"| {r['config'].replace('m6_','')} | {r['params']:,} | "
              f"{r['best_nats_mean']:.4f} +/- {r['best_nats_std']:.4f} | "
              f"{r['best_bpc_mean']:.4f} +/- {r['best_bpc_std']:.4f} | "
              f"{r['final_nats_mean']:.4f} +/- {r['final_nats_std']:.4f} | "
              f"{r['final_bpc_mean']:.4f} +/- {r['final_bpc_std']:.4f} |")

    print("\nBest seed per config (for word-validity):")
    for r in summary:
        print(f"  {r['config']}: seed {r['best_seed']} "
              f"(best val {r['best_seed_val']:.4f}) -> {r['best_seed_ckpt']}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nWrote summary JSON to {args.json}")


if __name__ == "__main__":
    main()
