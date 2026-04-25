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

The `e2e/` directory exists to reserve a clear place for full workflow tests. It may remain small or empty until the project has a true end-to-end test.

## Running Tests

Run the full suite from the repository root:

    pytest tests