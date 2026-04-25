# ADR: Use Local Checkpoint Files

## Status

Accepted

## Context

Training and generation happen at different times.

After training, the model needs a way to save what it learned. Generation then needs a way to load that saved model state.

For this project, a local file is enough. A database, model registry, or experiment tracking system would add too much complexity.

## Decision

Local checkpoint files are used to save and load model state.

Training writes checkpoints to disk. Generation rebuilds the model and loads a checkpoint before producing text.

## Consequences

This keeps the training and generation workflow simple.

It also makes the handoff between training and generation easy to inspect.

The tradeoff is that checkpoint management is basic. This is acceptable because the project is local and learning-focused.