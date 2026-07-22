# Quickstart

This guide walks through running **Tiny Transformer** on your own machine: installing it, training it on a small text file, and generating text from it.

## Prerequisites

- Python 3.11 or later, installed and available on your PATH (check with `python --version`)
- `pip`, which comes bundled with Python

Everything else (PyTorch, PyYAML, pytest) gets installed in step 2.

## 1. Create a virtual environment

A virtual environment keeps this project's dependencies separate from anything else on your machine.

From the repository root, run:

    python -m venv .venv

Then activate it. On Windows (PowerShell):

    .\.venv\Scripts\Activate.ps1

On macOS/Linux:

    source .venv/bin/activate

You'll know it worked if your terminal prompt now starts with `(.venv)`.

## 2. Install dependencies

    pip install -e .

This installs PyTorch, PyYAML, and pytest, everything the project needs.

## 3. Add training text

The model learns by reading a text file character by character, so it needs something to read. Create a file at:

    data/input.txt

and add plain text to it. Any text works: a short story, an article, lyrics, whatever you have on hand.

**One important rule:** the file needs to be long enough for the config you plan to train with. Each config sets a `context_window` (how many characters the model looks at when predicting the next one) and a `train_split_ratio` (how much of the file is used for training versus validation). The validation portion has to contain more characters than the context window, or training will fail with an error like `Token ID list must be longer than the context window`.

As a rule of thumb, your file's total length should be at least:

    context_window / (1 - train_split_ratio)

For the debug config below (`context_window: 32`, `train_split_ratio: 0.9`), that's about **330 characters** minimum. For the base config (`context_window: 128`), it's closer to **1,300 characters**.

Feel free to use any text you have on hand, just keep the length rule above in mind.

## 4. Pick a config

Configs live in `configs/` and control model size, training settings, and more.

- `configs/base.yaml` is the full-size config.
- `configs/debug.yaml` is smaller and trains much faster, good for a first run or a smoke test.

Start with the debug config.

## 5. Train the model

## 5. Train the model

    python scripts/train.py --config configs/debug.yaml

This reads your training text, builds the model, trains it, and saves a checkpoint. Where checkpoints are written depends on the config's `checkpointing.output_dir`. For the debug config, that's:

    outputs/checkpoints-debug/

(the base config uses `outputs/checkpoints/` instead)

## 6. Generate text

Once training finishes, generate text from the saved checkpoint:

    python scripts/generate.py --config configs/debug.yaml --prompt "hello"

This loads the checkpoint, rebuilds the tokenizer from `data/input.txt`, and generates new text starting from your prompt.

## 7. Run the tests (optional)

To confirm everything is working as expected:

    pytest tests

Tests are split into `tests/unit/`, `tests/integration/`, and `tests/e2e/`.

## Notes

This project is intentionally small: a learning tool for understanding how a transformer works, not a production system. Don't expect polished output, especially from the debug config on a short text file.

For more detail:

- `README.md` for the project overview
- `ARCHITECTURE.md` for the architecture explanation
- `docs/adr/` for architecture decision records