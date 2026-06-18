"""Download and place the TinyShakespeare corpus into data/.

TinyShakespeare is about 1 MB of plain text (the concatenated works in a single
file). It is the canonical small corpus for educational language models, which
makes Picoder results easy to compare against prior art.

Usage:
    python prepare_data.py
    python prepare_data.py --data-dir data --dataset tinyshakespeare
"""

from __future__ import annotations

import argparse
import os
import urllib.request

# Karpathy's char-rnn mirror of the Tiny Shakespeare dataset.
TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
    "tinyshakespeare/input.txt"
)


def download(url: str, dest: str) -> None:
    print(f"Downloading {url}")
    with urllib.request.urlopen(url) as resp:
        data = resp.read()
    with open(dest, "wb") as f:
        f.write(data)
    print(f"Wrote {len(data):,} bytes to {dest}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the Picoder corpus.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--dataset", default="tinyshakespeare")
    parser.add_argument(
        "--force", action="store_true", help="re-download even if the file exists"
    )
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    dest = os.path.join(args.data_dir, f"{args.dataset}.txt")

    if os.path.exists(dest) and not args.force:
        size = os.path.getsize(dest)
        print(f"{dest} already exists ({size:,} bytes). Use --force to re-download.")
        return

    if args.dataset == "tinyshakespeare":
        download(TINY_SHAKESPEARE_URL, dest)
    else:
        raise SystemExit(
            f"Unknown dataset '{args.dataset}'. Only 'tinyshakespeare' is built in."
        )

    # Quick sanity report so the corpus details are visible at a glance.
    with open(dest, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"Characters: {len(text):,}")
    print(f"Unique characters (vocab size): {len(set(text))}")
    print(f"First 120 chars:\n{text[:120]!r}")


if __name__ == "__main__":
    main()
