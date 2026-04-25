# Test Suite

The test suite is organized by test type.

## Structure

| Directory | Purpose |
|---|---|
| `unit/` | Focused tests for one module, class, or function at a time. |
| `integration/` | Tests that combine multiple project components or touch real project files, configs, checkpoints, or higher-level workflows. |
| `e2e/` | End-to-end tests that exercise a full user workflow, such as training from config, saving a checkpoint, and generating from that checkpoint. |
| `helpers/` | Shared test support code. This directory is not a test category by itself. |

## Current Notes

Most tests are unit or integration tests.

The `e2e/` directory contains smoke-style tests for full project workflows. These tests should stay small and should verify that major pieces work together, not exhaustively retest every module.

## Running Tests

Run the full suite from the repository root:

    pytest tests