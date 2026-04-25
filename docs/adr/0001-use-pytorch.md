# ADR: Use PyTorch

## Status

Accepted

## Context

This project is about learning how a small transformer works.

The project needs a tool that can handle tensors, neural network layers, gradients, and training. The code also needs to stay readable.

PyTorch is a common machine learning library. It provides the needed tools without hiding too much of the model code.

## Decision

PyTorch was chosen for the model, training loop, tensors, gradients, and optimizer support.

The transformer parts are still implemented directly instead of using a ready-made transformer model.

## Consequences

This makes the project easier to build and test.

PyTorch handles the low-level math and gradient work, so the project can focus on how the transformer is put together.

The tradeoff is that the project depends on a large machine learning library. This is acceptable because writing tensor operations and automatic gradients from scratch would be too much for this project.