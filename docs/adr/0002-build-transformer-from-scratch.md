# ADR: Build Transformer From Scratch

## Status

Accepted

## Context

This project is meant to help explain how a transformer works.

There are libraries, such as Hugging Face Transformers, that already provide transformer models. Those libraries are powerful, but they hide many details.

For this project, the goal is not to use the fastest or most complete model library. The goal is to learn by building the important parts directly.

## Decision

The transformer model is built from scratch using PyTorch.

This includes embeddings, attention, masking, feedforward layers, residual connections, layer normalization, training, and generation.

A prebuilt Hugging Face transformer model is not used for the first version.

## Consequences

This makes the project better as a learning tool.

The code shows how the main pieces connect instead of hiding them behind a high-level library.

The tradeoff is that the project has fewer features than a mature library. It may also be slower and less flexible. That is acceptable because the goal is understanding, not production use.