# ADR: Use Decoder-Only Transformer

## Status

Accepted

## Context

This project is about generating text.

For text generation, the model predicts the next token based on the tokens before it.

A decoder-only transformer is a good fit for this task because it is designed for autoregressive generation.

## Decision

A decoder-only transformer architecture is used.

The model uses causal masking so each position can only look at earlier positions and itself.

## Consequences

This keeps the model focused on next-token prediction.

It also matches the kind of architecture used by many text generation models.

The tradeoff is that this project does not cover encoder-only models or encoder-decoder models. That is acceptable because those are outside the first version’s scope.