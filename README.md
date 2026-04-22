# Tiny Transformer From Scratch

A planned beginner-friendly PyTorch project for building a small decoder-only transformer for character-level language modeling.

The goal of this project is to learn how transformer-based text generation works by implementing the core parts step by step in a codebase that stays small, readable, and organized.

This is not meant to be a large-scale model or a production-ready system. It is meant to be a practical learning project that makes the main ideas behind autoregressive language models easier to understand through implementation.

## What This Project Is About

Transformers are a type of neural network architecture used for sequence modeling, including modern language models. A decoder-only transformer learns to predict the next token in a sequence based on the tokens that came before it.

In this project, the model works at the **character level**, meaning each character is treated as a token. That keeps the tokenizer and data pipeline simple, which makes it easier to focus on the core mechanics of the model itself.

The project is intended to cover the main parts of a small language model, including:

- tokenization
- embeddings
- masked self-attention
- feedforward layers
- residual connections and layer normalization
- training and validation
- autoregressive text generation
- checkpointing
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

Planned features for the initial version:

- character-level tokenization
- token and positional embeddings
- masked multi-head self-attention
- feedforward blocks
- residual connections and layer normalization
- training loop with validation
- autoregressive text generation
- checkpoint save/load
- configuration-driven hyperparameters
- basic tests

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

This project is planned as a learning exercise, so the design choices are meant to reduce complexity where possible.

A few examples:

- **character-level tokenization** keeps preprocessing simple
- **decoder-only architecture** keeps the modeling task focused on next-token prediction
- **small repository structure** helps avoid one large script that becomes hard to maintain
- **configuration-driven settings** make it easier to experiment without scattering constants through the code

The structure described here is a plan, not a claim that every implementation detail is already known. Some parts will likely change during development. The purpose of planning the project this way is to give the work a clear direction and keep it manageable while learning.

## Planned Repository Structure

```text
tiny-transformer/
├── README.md
├── ARCHITECTURE.md
├── configs/
├── data/
├── docs/
│   └── adr/
├── src/
│   └── tiny_transformer/
│       ├── data/
│       ├── model/
│       ├── training/
│       ├── inference/
│       └── utils/
├── scripts/
└── tests/
```

This layout is meant to separate concerns early and make the code easier to navigate. It may change as the project develops.

## Planned Architecture

The codebase will be organized into a few main areas:

- **data**: dataset loading, splitting, tokenization
- **model**: embeddings, attention, feedforward layers, transformer blocks
- **training**: loss, optimization, checkpoints, training loop
- **inference**: prompt handling and text generation
- **utils/config**: shared helpers and configuration

The goal is to keep modules focused and avoid tightly coupling everything into one file.

## Engineering Principles

This project is being built with a few simple principles in mind:

- no hidden magic numbers
- clear module boundaries
- minimal duplication
- configuration over scattered constants
- readable and testable code
- simple design before clever design

These are goals for how the project should evolve, not a claim that the first version will be perfect.

## Planned Documentation

The repository is expected to include:

- `README.md` for the project overview and usage
- `ARCHITECTURE.md` for the technical structure
- a small set of ADRs for key design decisions

Possible early ADRs:

- use character-level tokenization for v1
- use a decoder-only architecture
- separate training and generation paths

## Planned Initial Milestones

| Milestone | Component | Status |
|---|---|---|
| **Milestone 1** | **Overall milestone status** | **Complete** |
| Milestone 1 | Repository scaffold | Done |
| Milestone 1 | Config setup | Done |
| Milestone 1 | Tokenizer | Done |
| Milestone 1 | Dataset pipeline | Done |
| **Milestone 2** | **Overall milestone status** | **Pending** |
| Milestone 2 | Transformer model | Pending |
| Milestone 2 | Masking | Pending |
| Milestone 2 | Forward-pass tests | Pending |
| **Milestone 3** | **Overall milestone status** | **Pending** |
| Milestone 3 | Training loop | Pending |
| Milestone 3 | Validation | Pending |
| Milestone 3 | Checkpointing | Pending |
| **Milestone 4** | **Overall milestone status** | **Pending** |
| Milestone 4 | Text generation | Pending |
| Milestone 4 | Sampling | Pending |
| Milestone 4 | Smoke tests | Pending |
| **Milestone 5** | **Overall milestone status** | **Pending** |
| Milestone 5 | Documentation | Pending |
| Milestone 5 | ADRs | Pending |
| Milestone 5 | Cleanup and polish | Pending |

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

## Status

Planned
