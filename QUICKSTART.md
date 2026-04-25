# Quickstart

This guide explains how to run **Tiny Transformer From Scratch** locally.

The project trains a small decoder-only transformer on a local text file and then generates text from a saved checkpoint.

## 1. Create a virtual environment

From the repository root:

    python -m venv .venv

Activate it in PowerShell:

    .\.venv\Scripts\Activate.ps1

## 2. Install dependencies

Install the project dependencies:

    pip install -e .

If test dependencies are not installed yet, install `pytest` as well:

    pip install pytest

## 3. Add training text

Create a file named:

    data/input.txt

Add plain text to that file. This text is the training corpus.

Example:

    hello world
    this is a tiny transformer training example

The model uses character-level tokenization, so it learns from the characters in this file.

## 4. Review the config

The default config is:

    configs/base.yaml

For a faster local run, use:

    configs/debug.yaml

The debug config is smaller and is better for smoke testing.

## 5. Train the model

Run training with the debug config:

    python scripts/train.py --config configs/debug.yaml

Training saves checkpoints to the configured checkpoint output directory.

By default, checkpoints are written under:

    outputs/checkpoints/

## 6. Generate text

After training, run generation from a saved checkpoint:

    python scripts/generate.py --config configs/debug.yaml --prompt "hello"

The script rebuilds the tokenizer from `data/input.txt`, loads the checkpoint, and generates text from the prompt.

## 7. Run tests

Run the full test suite:

    pytest tests

The tests are organized into:

    tests/unit/
    tests/integration/
    tests/e2e/

## Notes

This project is intentionally small. It is meant for learning how a transformer language model works, not for production training or large-scale text generation.

For more detail, see:

- `README.md` for the project overview
- `ARCHITECTURE.md` for the architecture explanation
- `docs/adr/` for architecture decision records