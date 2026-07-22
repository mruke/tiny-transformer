# Tiny Transformer

A small PyTorch project that implements a decoder-only transformer for character-level language modeling, built from scratch to learn how transformer-based text generation actually works.

This is a learning project, not a production system or a large-scale model. The design favors readable, modular, well-tested code over model scale or training performance. The goal is to make the mechanics of an autoregressive language model concrete by building each piece: tokenization, embeddings, masked self-attention, feedforward layers, residual connections and layer normalization, training and validation, checkpointing, generation, and sampling.

Character-level tokenization and a decoder-only architecture keep the scope small enough to implement and understand end to end, while still covering the core ideas behind modern text generation.

## Scope

Implemented in this version:

- character-level tokenization
- token and positional embeddings
- masked multi-head self-attention
- feedforward blocks
- residual connections and layer normalization
- training loop with validation
- checkpoint save/load
- autoregressive text generation (greedy, temperature, top-k sampling)
- configuration-driven hyperparameters
- focused tests

Out of scope for this version:

- large-scale or distributed training
- mixed precision
- advanced (subword) tokenization
- production inference serving
- optimization-heavy attention kernels

Possible follow-ups: KV-cache support, subword tokenizer support, benchmarking/profiling, richer evaluation, inference optimization experiments.

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

This layout separates configuration, data handling, model code, training, inference, scripts, and tests. See [`ARCHITECTURE.md`](ARCHITECTURE.md) for module responsibilities, runtime flows, diagrams, and design tradeoffs, and `docs/adr/` for individual design decisions.

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

- trains end to end on a small text dataset
- generates text from a saved checkpoint
- runs with a clean, understandable structure
- exposes key hyperparameters through configuration
- passes a basic test suite
- includes architecture and design documentation

## References and Acknowledgements

Built as a learning-focused implementation of a small decoder-only transformer, informed by:

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762), the original transformer paper.
- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/), an educational PyTorch walkthrough.
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/), a visual explanation of attention.
- [PyTorch Tutorials](https://docs.pytorch.org/tutorials/), official PyTorch learning materials.

ChatGPT was used during development as a learning, design, and review assistant for project planning, architecture discussion, documentation drafting, implementation review, and test-suite design. Some tests were generated with ChatGPT assistance and then reviewed, adapted, and validated as part of this repository.

## License

MIT. See [`LICENSE`](LICENSE).

## Status

Version 0.5.0 complete. The first portfolio-ready version is finished.


