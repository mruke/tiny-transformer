# ADR: Separate Training and Inference Workflows

## Status

Accepted

## Context

Training and generation are related, but they are not the same workflow.

Training reads data, builds batches, updates model weights, validates the model, and saves checkpoints.

Generation loads a checkpoint, takes a prompt, predicts new tokens, and decodes text.

Putting both workflows into one large file would make the project harder to read.

## Decision

Training and inference are kept as separate workflows.

Training code lives in the training package and uses `scripts/train.py`.

Generation code lives in the inference package and uses `scripts/generate.py`.

## Consequences

This makes the project easier to understand.

Each workflow has a clear job.

The tradeoff is that there are more files. This is acceptable because the files have separate responsibilities and are easier to test.