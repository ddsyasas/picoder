"""Word-validity metric for Picoder samples.

Makes the qualitative "structure before fluency" claim quantitative: generate a
fixed amount of text from a checkpoint, split into whitespace tokens, strip
surrounding punctuation, lowercase, and report the fraction that are real English
words according to a wordlist. A character-level model with no notion of words
will still emit many valid short words by chance; the fraction rising with model
quality is the signal of interest.

Generation is deterministic given the seed (default seed 1337, temperature 0.5,
top_k 40, fixed prompt) so the number is reproducible and comparable across
checkpoints.

Usage:
    python word_validity.py --checkpoint checkpoints/pico/best.pt
    python word_validity.py --checkpoint checkpoints/pico/best.pt --max-new-tokens 5000
"""

from __future__ import annotations

import argparse
import os
import string
import urllib.request

from src import load

# A common English wordlist mirror, used only if the system dictionary is absent.
WORDLIST_URL = (
    "https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt"
)
WORDLIST_CACHE = "data/words_alpha.txt"


def load_wordlist() -> set[str]:
    """Load a set of lowercase English words.

    Prefers the system dictionary (/usr/share/dict/words); otherwise downloads a
    common wordlist once and caches it under data/.
    """
    path = "/usr/share/dict/words"
    if not os.path.exists(path):
        if not os.path.exists(WORDLIST_CACHE):
            os.makedirs("data", exist_ok=True)
            print(f"Downloading wordlist from {WORDLIST_URL}")
            urllib.request.urlretrieve(WORDLIST_URL, WORDLIST_CACHE)
        path = WORDLIST_CACHE
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return {line.strip().lower() for line in f if line.strip()}


def word_validity(text: str, words: set[str]) -> tuple[float, int, int]:
    """Return (fraction_valid, n_valid, n_tokens) for generated text.

    Tokens are whitespace-split, stripped of surrounding punctuation, lowercased,
    and required to be non-empty. A token counts as valid if it is in the
    wordlist.
    """
    tokens = []
    for raw in text.split():
        t = raw.strip(string.punctuation + string.digits).lower()
        if t:
            tokens.append(t)
    if not tokens:
        return 0.0, 0, 0
    n_valid = sum(1 for t in tokens if t in words)
    return n_valid / len(tokens), n_valid, len(tokens)


def main() -> None:
    parser = argparse.ArgumentParser(description="Word-validity of Picoder samples.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=5000)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--prompt", default="\n")
    args = parser.parse_args()

    words = load_wordlist()
    model = load(args.checkpoint)
    text = model.generate(
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
    )
    frac, n_valid, n_tokens = word_validity(text, words)
    print(f"checkpoint: {args.checkpoint}")
    print(f"generated chars: {len(text)}  tokens: {n_tokens}  valid words: {n_valid}")
    print(f"word-validity: {frac*100:.1f}%  ({n_valid}/{n_tokens})")


if __name__ == "__main__":
    main()
