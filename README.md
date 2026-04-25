# Tiny Transformer From Scratch

A small PyTorch project for building a decoder-only transformer from scratch for character-level language modeling.

The goal of this project is to learn how transformer-based text generation works by implementing the core parts step by step in a codebase that stays small, readable, and organized.

This is not meant to be a large-scale model or a production-ready system. It is meant to be a practical learning project that makes the main ideas behind autoregressive language models easier to understand through implementation.

## What This Project Is About

Transformers are a type of neural network architecture used for sequence modeling, including modern language models. A decoder-only transformer learns to predict the next token in a sequence based on the tokens that came before it.

In this project, the model works at the **character level**, meaning each character is treated as a token. That keeps the tokenizer and data pipeline simple, which makes it easier to focus on the core mechanics of the model itself.

The project covers the main parts of a small language model, including:

- tokenization
- embeddings
- masked self-attention
- feedforward layers
- residual connections and layer normalization
- training and validation
- checkpointing
- autoregressive text generation
- sampling strategies
- basic testing

## Purpose

The purpose of this implementation is to better understand how a transformer language model works by building one from scratch.

Rather than only using high-level libraries or pretrained models, this project is meant to make the individual pieces more concrete. Building a small model end to end helps connect the theory to actual code and makes the training and generation process easier to reason about.

Keeping the project small is intentional. A character-level decoder-only model is limited enough to be manageable, but still includes the important ideas behind text generation with transformers.

This project is also meant to practice writing ML code in a clean and maintainable way. Part of the goal is not just to make the model run, but to structure the repository so that the responsibilities of data handling, model code, training, and inference are separated clearly.

## Goals

The first version of this project is meant to:

- implement a tiny decoder-only transformer from scratch
- train on a small text corpus
- generate text autoregressively from a prompt
- keep the code modular and easy to follow
- document major architecture and design decisions

## Scope

Implemented features for the initial version:

- character-level tokenization
- token and positional embeddings
- masked multi-head self-attention
- feedforward blocks
- residual connections and layer normalization
- training loop with validation
- checkpoint save/load
- autoregressive text generation
- greedy sampling
- temperature sampling
- top-k filtering
- configuration-driven hyperparameters
- focused tests

This first version is intentionally limited so the project stays understandable and realistic to complete.

## Non-Goals

The initial version is not intended to include:

- large-scale training
- distributed training
- mixed precision
- advanced tokenizer implementations
- production inference serving
- optimization-heavy attention kernels

These are possible future directions, but they are outside the scope of the first pass.

## Why Build It This Way

This project is built as a learning exercise, so the design choices are meant to reduce complexity where possible.

A few examples:

- **character-level tokenization** keeps preprocessing simple
- **decoder-only architecture** keeps the modeling task focused on next-token prediction
- **small repository structure** helps avoid one large script that becomes hard to maintain
- **configuration-driven settings** make it easier to experiment without scattering constants through the code

The project keeps the implementation small while still separating the major responsibilities of a transformer language modeling workflow.

## Repository Structure

    tiny-transformer/
    ├── README.md
    ├── ARCHITECTURE.md
    ├── pyproject.toml
    ├── configs/
    │   ├── base.yaml
    │   └── debug.yaml
    ├── data/
    ├── docs/
    │   └── adr/
    ├── scripts/
    │   ├── train.py
    │   └── generate.py
    ├── src/
    │   └── tiny_transformer/
    │       ├── config/
    │       ├── data/
    │       ├── model/
    │       ├── training/
    │       ├── inference/
    │       └── utils/
    └── tests/

This layout separates configuration, data handling, model code, training, inference, scripts, and tests.

## Architecture

The codebase is organized around separate packages for configuration, data handling, model implementation, training, inference, scripts, and tests.

For a more detailed architecture overview, including module responsibilities, runtime flows, diagrams, and design tradeoffs, see [`ARCHITECTURE.md`](ARCHITECTURE.md).

## Engineering Principles

The project favors readable, testable, modular code over clever abstractions or production-scale infrastructure.

## Documentation

Additional project documentation includes:

- [`ARCHITECTURE.md`](ARCHITECTURE.md) for the technical structure, runtime flows, and design tradeoffs
- `docs/adr/` for architecture decision records

## Milestones

| Milestone | Component | Status | Tag |
|---|---|---|---|
| **Milestone 1** | **Overall** | **Complete** | **v0.1.0** |
| Milestone 1 | Repository scaffold | Done | — |
| Milestone 1 | Config setup | Done | — |
| Milestone 1 | Tokenizer | Done | — |
| Milestone 1 | Dataset pipeline | Done | — |
| **Milestone 2** | **Overall** | **Complete** | **v0.2.0** |
| Milestone 2 | Transformer model | Done | — |
| Milestone 2 | Masking | Done | — |
| Milestone 2 | Forward-pass tests | Done | — |
| **Milestone 3** | **Overall** | **Complete** | **v0.3.0** |
| Milestone 3 | Training loop | Done | — |
| Milestone 3 | Validation | Done | — |
| Milestone 3 | Checkpointing | Done | — |
| **Milestone 4** | **Overall** | **Complete** | **v0.4.0** |
| Milestone 4 | Text generation | Done | — |
| Milestone 4 | Sampling | Done | — |
| Milestone 4 | Smoke tests | Done | — |
| **Milestone 5** | **Overall** | **Complete** | **v0.5.0** |
| Milestone 5 | Documentation | Done | — |
| Milestone 5 | ADRs | Done | — |
| Milestone 5 | Cleanup and polish | Done | — |

## Definition of Done for v1

The first version will be considered complete when it can:

- train end to end on a small text dataset
- generate text from a saved checkpoint
- run with a clean and understandable structure
- expose key hyperparameters through configuration
- pass a basic test suite
- include architecture and design documentation

## Future Extensions

Possible follow-up work after v1:

- KV-cache support
- subword tokenizer support
- benchmarking and profiling
- richer evaluation
- inference optimization experiments

## References and Acknowledgements

This project was built as a learning-focused implementation of a small decoder-only transformer for character-level language modeling.

The implementation was informed by public educational and research resources on transformers, attention, neural networks, and PyTorch. These references were used for learning and conceptual grounding.

Useful references include:

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762), the original transformer paper introducing the attention-based transformer architecture.
- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/), an educational PyTorch walkthrough of transformer building blocks.
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/), a visual explanation of transformer concepts and attention.
- [PyTorch Tutorials](https://docs.pytorch.org/tutorials/), official PyTorch learning materials for tensors, neural network modules, autograd, and training workflows.

ChatGPT was used during development as a learning, design, and review assistant. It helped with project planning, architecture discussion, documentation drafting, implementation review, and test-suite design. Some tests were generated with ChatGPT assistance and then reviewed, adapted, and validated as part of this repository.

## Status

Version 0.5.0 complete. The first portfolio-ready version is finished.


