# Architecture

This document gives a compact architecture overview for **Tiny Transformer From Scratch**.

The structure loosely follows the arc42 style, but only includes sections that are useful for a small local machine learning project. The goal is to document the important engineering decisions without turning the project into an over-formal architecture exercise.

---

## 1. Introduction and Goals

**Tiny Transformer From Scratch** is a small decoder-only transformer implemented in PyTorch for character-level language modeling.

The project is designed as a portfolio-quality software engineering artifact. It prioritizes readable implementation, modular structure, focused tests, and clear separation of responsibilities over model scale or training performance.

The main goals are:

- implement the core pieces of a decoder-only transformer from scratch
- keep the codebase small enough to understand end to end
- separate data, model, training, inference, and configuration concerns
- support local training, validation, checkpointing, and text generation
- demonstrate disciplined testing and maintainable project structure

Non-goals:

- production-scale model training
- distributed training
- advanced tokenizer implementations
- serving an API or web application
- maximizing benchmark performance

---

## 2. Constraints

The project is intentionally constrained to keep the architecture understandable.

| Constraint | Impact |
|---|---|
| Python and PyTorch | The model and training loop use familiar ML tooling while keeping implementation details visible. |
| Character-level modeling | Tokenization stays simple and transparent. |
| Decoder-only transformer | The architecture focuses on autoregressive language modeling. |
| Local execution | Training and generation run from scripts on a local machine. |
| Filesystem-based inputs and outputs | Configs, input text, and checkpoints are regular project files. |
| Small dependency footprint | The project avoids unnecessary framework or infrastructure dependencies. |
| Portfolio readability | Structure, naming, and tests should be easy for a reviewer to inspect. |

---

## 3. Solution Strategy

The system is organized around a small number of focused packages:

- `config` loads and validates runtime configuration
- `data` converts raw text into token sequences for training and validation
- `model` contains the transformer implementation
- `training` owns loss calculation, optimization, validation, and checkpointing
- `inference` owns autoregressive generation and sampling strategies
- `scripts` provide simple command-line entry points
- `tests` verify behavior at module boundaries

The design favors direct composition over heavy abstraction. Files are split when they represent genuinely separate responsibilities, not merely to create architectural ceremony.

Training and generation share the same core model and tokenizer concepts, but are exposed through separate scripts because they are separate user workflows.

---

## 4. System Context

The system is used by a developer or reviewer from a local development environment.

The user interacts with the project through command-line scripts. The scripts read configuration and input text from the local filesystem, run PyTorch code, and write checkpoints or generated text output.

There are no external services, databases, APIs, queues, or deployment infrastructure.

---

## 5. Container View

This project is not a distributed system, so the C4 "container" concept is used lightly. The main containers are the local scripts, the Python application code, the local filesystem, and the PyTorch runtime.

    +----------------------------------------------------------------------------------+
    | Tiny Transformer From Scratch                                                    |
    |----------------------------------------------------------------------------------|
    | Purpose: train and run a small decoder-only transformer for character-level LM    |
    +----------------------------------------------------------------------------------+

        [Developer / User]
                |
                v
    +-------------------------+        reads/writes        +--------------------------+
    | CLI Scripts             | <------------------------> | Local Filesystem         |
    |-------------------------|                            |--------------------------|
    | scripts/train.py        |                            | configs/*.yaml           |
    | scripts/generate.py     |                            | data/input.txt           |
    |                         |                            | outputs/checkpoints/*.pt |
    +-------------------------+                            +--------------------------+
                |
                v
    +----------------------------------------------------------------------------------+
    | Tiny Transformer Python Application                                              |
    |----------------------------------------------------------------------------------|
    | config      - configuration loading and validation                               |
    | data        - tokenizer, train/validation split, sequence dataset                |
    | model       - embeddings, attention, feedforward layers, transformer blocks      |
    | training    - losses, optimizer, metrics, trainer, checkpoints                  |
    | inference   - generation loop, generator, sampling strategies                   |
    +----------------------------------------------------------------------------------+
                |
                v
    +-------------------------+
    | PyTorch Runtime         |
    |-------------------------|
    | tensors                 |
    | autograd                |
    | neural network modules  |
    | optimization            |
    +-------------------------+

---

## 6. Building Block View

The application code is organized by responsibility rather than by technical layer alone.

| Area | Responsibility |
|---|---|
| `src/tiny_transformer/config` | Configuration schema, loading, validation, and configuration-specific errors. |
| `src/tiny_transformer/data` | Character tokenization, token splitting, and sequence dataset creation. |
| `src/tiny_transformer/model` | Transformer model components and top-level decoder-only model. |
| `src/tiny_transformer/training` | Training loop, validation, loss calculation, optimizer setup, metrics, and checkpoints. |
| `src/tiny_transformer/inference` | Generation validation, autoregressive generation loop, generator orchestration, and sampling. |
| `scripts` | User-facing training and generation entry points. |
| `tests` | Focused unit and integration-style checks for project behavior. |

### Component View

The component view zooms into the Python application package.

    +----------------------------------------------------------------------------------+
    | Tiny Transformer Python Application                                              |
    +----------------------------------------------------------------------------------+

    +------------------+       +------------------+       +---------------------------+
    | config           | ----> | data             | ----> | model                     |
    |------------------|       |------------------|       |---------------------------|
    | load config      |       | tokenizer        |       | token embeddings          |
    | validate config  |       | train/val split  |       | positional embeddings     |
    | config errors    |       | sequence dataset |       | causal self-attention     |
    +------------------+       +------------------+       | feedforward layers        |
                                                          | transformer blocks        |
                                                          | top-level transformer     |
                                                          +---------------------------+
                                                                      |
                                                                      v
                                     +------------------------------------------------+
                                     | training                                       |
                                     |------------------------------------------------|
                                     | losses                                         |
                                     | optimizer setup                                |
                                     | metrics                                        |
                                     | trainer                                        |
                                     | checkpoint save/load                           |
                                     +------------------------------------------------+
                                                                      |
                                                                      v
                                     +------------------------------------------------+
                                     | inference                                      |
                                     |------------------------------------------------|
                                     | generation validation                          |
                                     | generation loop                                |
                                     | generator                                      |
                                     | greedy sampling                                |
                                     | temperature sampling                           |
                                     | top-k filtering                                |
                                     +------------------------------------------------+

    Entry points:

    scripts/train.py
        -> config
        -> data
        -> model
        -> training
        -> checkpoint output

    scripts/generate.py
        -> config
        -> data/tokenizer reconstruction
        -> model
        -> checkpoint loading
        -> inference
        -> generated text output

---

## 7. Runtime View

### 7.1 Training Flow

The training script coordinates the end-to-end training workflow.

    1. Load configuration.
    2. Read the input corpus.
    3. Build the character tokenizer.
    4. Encode the text into token IDs.
    5. Split token IDs into training and validation ranges.
    6. Build sequence datasets.
    7. Instantiate the decoder-only transformer.
    8. Configure loss and optimizer.
    9. Train for the configured number of steps or epochs.
    10. Run validation.
    11. Save a checkpoint.

Important runtime boundary:

- `scripts/train.py` coordinates the workflow.
- The reusable implementation lives in package modules.
- Checkpoint persistence is handled by the training package, not by the model itself.

### 7.2 Generation Flow

The generation script loads a trained checkpoint and produces text autoregressively.

    1. Load configuration.
    2. Read the same corpus used to define the character vocabulary.
    3. Rebuild the tokenizer.
    4. Instantiate the model with matching configuration.
    5. Load model state from a checkpoint.
    6. Encode the prompt.
    7. Repeatedly predict the next token.
    8. Apply the selected sampling strategy.
    9. Append the selected token.
    10. Decode the final token sequence into text.

Supported generation strategies include:

- greedy sampling
- temperature sampling
- temperature sampling with top-k filtering

---

## 8. Deployment View

The project runs locally.

There is no separate deployment environment. The expected runtime environment is:

- a local Python environment
- project dependencies installed from project metadata
- local configuration files
- local input text
- local checkpoint output

The filesystem is the persistence boundary.

| Artifact | Role |
|---|---|
| `configs/*.yaml` | Runtime configuration. |
| `data/input.txt` | Training corpus input. |
| `outputs/checkpoints/*.pt` | Saved model checkpoints. |
| `scripts/train.py` | Training entry point. |
| `scripts/generate.py` | Generation entry point. |

---

## 9. Cross-Cutting Concepts

### Configuration

Configuration is loaded from YAML and validated before use. This keeps model, data, and training settings explicit and avoids scattering runtime constants throughout the codebase.

### Shape Discipline

Transformer code depends heavily on tensor shapes. Tests verify important shape behavior where it affects correctness, especially around attention, masks, model outputs, and generation.

### Causal Masking

The model uses causal masking so each position can only attend to earlier positions and itself. This is the core constraint that makes the transformer decoder suitable for autoregressive generation.

### Checkpointing

Checkpoints provide the bridge between training and generation. Training writes model state to disk; generation rebuilds the model structure and loads the saved state.

### Sampling

Sampling behavior is separated from the generator flow. This keeps generation orchestration readable and makes greedy, temperature, and top-k behavior independently testable.

### Testing

Tests are organized around focused behavior rather than one broad end-to-end test file. This makes failures easier to locate and keeps module responsibilities clear.

---

## 10. Architecture Decisions and Tradeoffs

### Character-level tokenizer

The project uses character-level tokenization because it is easy to inspect and implement from scratch. This keeps the tokenizer suitable for a learning-focused transformer project.

Tradeoff:

- simpler and more transparent
- less efficient and less capable than subword tokenization

### Decoder-only architecture

The model is decoder-only because the project focuses on autoregressive language modeling and text generation.

Tradeoff:

- well aligned with next-token prediction
- does not cover encoder-decoder translation or bidirectional encoder tasks

### Local scripts instead of a full CLI package

Training and generation are exposed through scripts rather than a packaged command-line application.

Tradeoff:

- simpler for a small portfolio project
- less polished than a dedicated installed CLI

### Modular sampling

Sampling strategies are split into focused modules.

Tradeoff:

- slightly more files
- clearer tests and easier future extension

### Minimal infrastructure

The project avoids experiment trackers, service layers, databases, and distributed execution.

Tradeoff:

- easier to understand and run locally
- not intended for production ML operations

---

## 11. Risks and Technical Debt

Known limitations:

- the model is intentionally small
- training is not optimized for speed or scale
- the tokenizer is simple and character-based
- generation usability is basic
- checkpoints are local files only
- there is no experiment tracking system
- there is no packaging beyond project metadata and scripts

These are acceptable for the project scope. The repository is intended to demonstrate clear implementation of a small transformer system, not production ML infrastructure.

---

## 12. Summary

The architecture is intentionally modest.

The important qualities are:

- clear module boundaries
- readable transformer implementation
- explicit training and generation workflows
- simple local persistence through checkpoints
- focused tests around important behavior
- minimal infrastructure

This keeps the project understandable while still demonstrating practical software engineering structure around a working PyTorch language model.