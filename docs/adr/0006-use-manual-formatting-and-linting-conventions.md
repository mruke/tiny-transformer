# ADR: Use Manual Formatting and Linting Conventions

## Status

Accepted

## Context

Many Python projects use tools like Black, Ruff, Flake8, or isort to format and check code.

Those tools are useful, but this project is small. The main goal is learning how the transformer works and keeping the code easy to read.

Adding formatting and linting tools would add more setup and more project rules.

## Decision

Manual formatting and simple code style conventions are used for the first version.

The code is kept readable through clear names, small functions, focused modules, and consistent formatting.

A full formatting or linting toolchain is not added yet.

## Consequences

This keeps the project setup simple.

It also keeps attention on the model, training, testing, and generation code.

The tradeoff is that formatting is not automatically enforced. This means code review and careful editing matter more. If the project grows, adding a tool like Ruff or Black may become useful later.