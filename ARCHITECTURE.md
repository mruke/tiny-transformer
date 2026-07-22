# Architecture

This document gives a architecture overview for **tiny-transformer**, loosely following the arc42 style, scoped to relevance for a small local machine learning project.

For project purpose and scope, see [`README.md`](README.md). For individual design decisions and their tradeoffs, see [`docs/adr/`](docs/adr/).

---

## 1. Constraints

The project is intentionally constrained to keep the architecture understandable.

| Constraint | Impact |
|---|---|
| Python and PyTorch | The model and training loop use familiar ML tooling while keeping implementation details visible. |
| Character-level modeling | Tokenization stays simple and transparent. |
| Decoder-only transformer | The architecture focuses on autoregressive language modeling. |
| Local execution | Training and generation run from scripts on a local machine; no external services, databases, APIs, or deployment infrastructure. |
| Filesystem-based inputs and outputs | Configs, input text, and checkpoints are regular project files. |
| Small dependency footprint | The project avoids unnecessary framework or infrastructure dependencies. |
| Portfolio readability | Structure, naming, and tests should be easy for a reviewer to inspect. |

---

## 2. Solution Strategy

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

## 3. Component View

A developer or reviewer interacts with the system through the CLI scripts, which read configuration and input text from the local filesystem, run PyTorch code, and write checkpoints or generated text output. There is no other runtime environment to consider.

    +----------------------------------------------------------------------------------+
    | tiny-transformer                                                    |
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
    | tiny-transformer Python Application                                              |
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

    Entry points:

    scripts/train.py
        -> config -> data -> model -> training -> checkpoint output

    scripts/generate.py
        -> config -> data/tokenizer reconstruction -> model
        -> checkpoint loading -> inference -> generated text output

---

## 4. Runtime View

### 4.1 Training Flow

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

`scripts/train.py` coordinates this workflow; the reusable implementation lives in the package modules, and checkpoint persistence is handled by the training package, not the model itself.

### 4.2 Generation Flow

1. Load configuration.
2. Read the same corpus used to define the character vocabulary.
3. Rebuild the tokenizer.
4. Instantiate the model with matching configuration.
5. Load model state from a checkpoint.
6. Encode the prompt.
7. Repeatedly predict the next token.
8. Apply the selected sampling strategy (greedy, temperature, or temperature with top-k).
9. Append the selected token.
10. Decode the final token sequence into text.

---

## 5. Cross-Cutting Concepts

**Configuration.** Loaded from YAML and validated before use, keeping model, data, and training settings explicit instead of scattering runtime constants throughout the codebase.

**Shape discipline.** Transformer code depends heavily on tensor shapes. Tests verify shape behavior where it affects correctness, especially around attention, masks, model outputs, and generation.

**Causal masking.** Each position can only attend to earlier positions and itself. This is the core constraint that makes the transformer decoder suitable for autoregressive generation.

**Checkpointing.** The bridge between training and generation: training writes model state to disk, and generation rebuilds the model structure and loads the saved state.

**Sampling.** Kept separate from the generator flow, so generation orchestration stays readable and greedy, temperature, and top-k behavior are independently testable.

**Testing.** Organized around focused behavior rather than one broad end-to-end file, so failures are easy to locate and module responsibilities stay clear.

---

## 6. Design Decisions

Major design decisions (PyTorch, character-level tokenization, decoder-only architecture, YAML configuration, local checkpoint files, and others) are recorded individually in [`docs/adr/`](docs/adr/), including context, decision, and consequences for each. This document describes the resulting structure; the ADRs record why it was chosen.

---

## 7. Risks and Technical Debt

Known limitations, acceptable for the project's scope:

- the model is intentionally small and training is not optimized for speed or scale
- the tokenizer is simple and character-based
- generation usability is basic
- checkpoints are local files only, with no experiment tracking system
- there is no packaging beyond project metadata and scripts

---

## 8. Glossary

| Term | Meaning |
|---|---|
| Autoregressive generation | A generation process where the model predicts one token at a time and then uses that new token as part of the next input. |
| Batch | A group of training examples processed together in one model step. |
| Causal mask | A mask that prevents a token from attending to future tokens. This keeps generation left-to-right. |
| Character-level tokenization | A tokenization approach where each unique character is treated as a token. |
| Checkpoint | A saved model state that can be loaded later for training continuation or generation. |
| Config | A settings file or object that controls project behavior, such as model size, training settings, and generation settings. |
| Context window | The number of previous tokens the model can look at when predicting the next token. |
| Corpus | The source text used to train the model. |
| Dataset | A structured collection of examples used for training or validation. |
| Decoder-only transformer | A transformer architecture that predicts future tokens from past tokens. It is commonly used for text generation. |
| Dropout | A training technique that randomly disables some values during training to reduce overfitting. |
| Embedding | A learned numeric representation of a token or position. |
| Epoch | One full pass through the training data. |
| Feedforward layer | A neural network layer inside each transformer block that processes each token representation after attention. |
| Generation | The process of producing new text from a prompt. |
| Gradient | A value used during training that shows how a model parameter should change to reduce loss. |
| Greedy sampling | A generation strategy that always chooses the highest-scoring next token. |
| Hyperparameter | A setting chosen before training, such as learning rate, batch size, or number of layers. |
| Inference | Running a trained model to make predictions or generate text. |
| Layer normalization | A normalization step that helps keep training stable inside the transformer. |
| Learning rate | A training setting that controls how large each optimizer update is. |
| Logits | Raw model output scores before they are converted into probabilities. |
| Loss | A number that measures how wrong the model prediction is. Lower loss usually means better predictions. |
| Masked self-attention | Self-attention with a mask applied so tokens cannot use information from future positions. |
| Model parameter | A learned value inside the model, such as a weight in a neural network layer. |
| Multi-head self-attention | A form of attention that lets the model look at token relationships in multiple learned ways at the same time. |
| Neural network | A model made of layers with learned parameters. It maps inputs to outputs through numeric operations. |
| Optimizer | The training component that updates model parameters using gradients. |
| Positional embedding | A learned representation that tells the model where each token appears in the sequence. |
| Prompt | The starting text given to the model before generation begins. |
| PyTorch | The machine learning library used by this project for tensors, neural network modules, gradients, and optimization. |
| Residual connection | A connection that adds a layer's input back to its output. This helps information and gradients flow through deep models. |
| Sampling | The process of choosing the next token from model output scores. |
| Seed | A value used to make random behavior more repeatable. |
| Self-attention | A mechanism that lets each token compare itself with other tokens in the same sequence. |
| Sequence | An ordered list of tokens. |
| Tensor | A multi-dimensional array of numbers used by PyTorch. Scalars, vectors, matrices, and batches can all be represented as tensors. |
| Token | A numeric unit of text used by the model. In this project, tokens represent characters. |
| Token ID | The integer assigned to a token by the tokenizer. |
| Tokenizer | The component that converts text into token IDs and token IDs back into text. |
| Top-k filtering | A sampling method that keeps only the highest-scoring `k` tokens before choosing the next token. |
| Training | The process of updating model parameters so the model becomes better at predicting the next token. |
| Training split | The part of the data used to update model parameters. |
| Validation | The process of measuring model performance on data not used for parameter updates. |
| Validation split | The part of the data used to check model performance during or after training. |
| Vocabulary | The set of tokens the model knows how to process. |
| Weight | A learned numeric parameter inside a neural network. |