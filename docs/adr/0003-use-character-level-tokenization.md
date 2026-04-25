# ADR: Use Character-Level Tokenization

## Status

Accepted

## Context

A language model needs to turn text into tokens before training.

More advanced models often use word pieces or subword tokenizers. These are useful, but they add more complexity.

This project is small and focused on learning the transformer itself. A character-level tokenizer is easier to understand and test.

## Decision

Character-level tokenization is used for the first version of the project.

Each unique character in the text becomes a token.

## Consequences

This keeps the tokenizer simple.

It makes the data pipeline easier to inspect because the mapping between text and token IDs is clear.

The tradeoff is that character-level models are less efficient than subword models. They usually need more steps to learn strong language patterns. This is acceptable for a small learning project.