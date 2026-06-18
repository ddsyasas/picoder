# Picoder

Picoder is a research project that builds a series of tiny, fully transparent
language models from scratch. The aim is to understand, document, and reproduce
every part of a working LLM at the smallest practical scale, then scale up step
by step across versions.

This is **Picoder 0.1**, the first build.

## What this is

A from-scratch, decoder-only transformer (GPT style) small enough to train on a
single consumer GPU, and even on CPU at the smallest configuration. It is built
for learning and research, with every design decision written down rather than
assumed.

## Status

Version 0.1 is in active development. The first target is a character-level
tokenizer trained on a small text corpus (TinyShakespeare), with a working
training loop and text generation.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Quick start

```bash
# 1. Download the corpus (TinyShakespeare, ~1 MB) into data/
python prepare_data.py

# 2. Train the default "pico" config (CPU is fine for a smoke test)
python train.py --config configs/pico.yaml

# 3. Generate text from a saved checkpoint
python sample.py --checkpoint checkpoints/pico/latest.pt --prompt "ROMEO:" --max-new-tokens 300
```

## Layout

```
prepare_data.py   download + place the corpus
train.py          training entry point
sample.py         text generation entry point
configs/          run configs (pico.yaml is the 0.1 default)
src/              library code: config, tokenizer, data, model
docs/             publishable documentation
data/             datasets (gitignored, created by prepare_data.py)
checkpoints/      saved weights (gitignored)
```

See `docs/` for the architecture overview and design notes.

## Versions

- **0.1**: minimal character-level transformer, single corpus, baseline training loop and sampler.
- **0.2 and beyond**: subword (BPE) tokenizer, larger corpora, and scaling experiments.

Each version is a tag in this same repository (v0.1, v0.2, ...), not a separate repo.

## License

To be decided before the first public release (MIT is a reasonable default for
a research artifact).

## Acknowledgements

Built in the spirit of small educational transformers such as the original GPT
line and Karpathy's nanoGPT. Prior-art and citation notes are kept with the
project research notes.
