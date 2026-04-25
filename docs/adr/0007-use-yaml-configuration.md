# ADR: Use YAML Configuration

## Status

Accepted

## Context

This project has settings for data, model size, training, generation, and checkpoints.

These settings could be hardcoded in Python files, but that would make experiments harder to change and harder to inspect.

A small project still benefits from having important settings in one clear place.

## Decision

YAML configuration files are used for project settings.

The configuration is loaded and validated before the settings are used by training or generation code.

## Consequences

This keeps important settings visible and easy to change.

It also avoids scattering constants through the codebase.

The tradeoff is that the project needs config loading and validation code. This is acceptable because it makes the project clearer and easier to adjust.