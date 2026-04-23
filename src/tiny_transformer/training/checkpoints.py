from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from tiny_transformer.config import AppConfig


# ---------------------------------------------------------------------------
# CheckpointMetadata
#
# CheckpointMetadata stores the small metadata fields saved with one
# checkpoint.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CheckpointMetadata:
    epoch: int
    project_name: str
    project_version: str


# ---------------------------------------------------------------------------
# _validate_model
#
# This function checks that the model is a PyTorch module.
# ---------------------------------------------------------------------------
def _validate_model(model: nn.Module) -> None:
    if not isinstance(model, nn.Module):
        raise TypeError("model must be an instance of torch.nn.Module.")


# ---------------------------------------------------------------------------
# _validate_optimizer
#
# This function checks that the optimizer is a PyTorch optimizer.
# ---------------------------------------------------------------------------
def _validate_optimizer(optimizer: Optimizer) -> None:
    if not isinstance(optimizer, Optimizer):
        raise TypeError("optimizer must be an instance of torch.optim.Optimizer.")


# ---------------------------------------------------------------------------
# _validate_config
#
# This function checks that config is an AppConfig instance.
# ---------------------------------------------------------------------------
def _validate_config(config: AppConfig) -> None:
    if not isinstance(config, AppConfig):
        raise TypeError("config must be an instance of AppConfig.")


# ---------------------------------------------------------------------------
# _validate_epoch
#
# This function checks that epoch is a positive integer.
# ---------------------------------------------------------------------------
def _validate_epoch(epoch: int) -> None:
    if not isinstance(epoch, int):
        raise TypeError("epoch must be an integer.")

    if epoch <= 0:
        raise ValueError("epoch must be a positive integer.")


# ---------------------------------------------------------------------------
# _validate_checkpoint_path
#
# This function checks that checkpoint path is a non-empty path-like value.
# ---------------------------------------------------------------------------
def _validate_checkpoint_path(checkpoint_path: str | Path) -> Path:
    if isinstance(checkpoint_path, Path):
        return checkpoint_path

    if isinstance(checkpoint_path, str) and checkpoint_path.strip():
        return Path(checkpoint_path)

    raise ValueError("checkpoint_path must be a non-empty path.")


# ---------------------------------------------------------------------------
# _validate_checkpoint_prefix
#
# This function checks that the checkpoint file prefix is a non-empty string.
# ---------------------------------------------------------------------------
def _validate_checkpoint_prefix(checkpoint_prefix: str) -> None:
    if not isinstance(checkpoint_prefix, str):
        raise TypeError("checkpoint_prefix must be a string.")

    if not checkpoint_prefix.strip():
        raise ValueError("checkpoint_prefix must be a non-empty string.")


# ---------------------------------------------------------------------------
# _ensure_checkpoint_parent_directory
#
# This function creates the parent directory for a checkpoint file.
# ---------------------------------------------------------------------------
def _ensure_checkpoint_parent_directory(checkpoint_path: Path) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# _build_checkpoint_metadata
#
# This function builds metadata saved inside one checkpoint.
# ---------------------------------------------------------------------------
def _build_checkpoint_metadata(
    config: AppConfig,
    epoch: int,
) -> CheckpointMetadata:
    _validate_config(config)
    _validate_epoch(epoch)

    return CheckpointMetadata(
        epoch=epoch,
        project_name=config.project.name,
        project_version=config.project.version,
    )


# ---------------------------------------------------------------------------
# build_checkpoint_path
#
# This function builds the checkpoint file path for one epoch.
# ---------------------------------------------------------------------------
def build_checkpoint_path(
    output_dir: str | Path,
    checkpoint_prefix: str,
    epoch: int,
) -> Path:
    resolved_output_dir = _validate_checkpoint_path(output_dir)
    _validate_checkpoint_prefix(checkpoint_prefix)
    _validate_epoch(epoch)

    checkpoint_filename = f"{checkpoint_prefix}_epoch_{epoch}.pt"

    return resolved_output_dir / checkpoint_filename


# ---------------------------------------------------------------------------
# _build_checkpoint_payload
#
# This function builds the checkpoint payload written to disk.
# ---------------------------------------------------------------------------
def _build_checkpoint_payload(
    model: nn.Module,
    optimizer: Optimizer,
    metadata: CheckpointMetadata,
) -> dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metadata": {
            "epoch": metadata.epoch,
            "project_name": metadata.project_name,
            "project_version": metadata.project_version,
        },
    }


# ---------------------------------------------------------------------------
# _validate_checkpoint_payload
#
# This function checks that a loaded checkpoint payload has the required
# structure.
# ---------------------------------------------------------------------------
def _validate_checkpoint_payload(checkpoint_payload: object) -> None:
    if not isinstance(checkpoint_payload, dict):
        raise ValueError("checkpoint payload must be a dictionary.")

    required_top_level_keys = (
        "model_state_dict",
        "optimizer_state_dict",
        "metadata",
    )

    for required_key in required_top_level_keys:
        if required_key not in checkpoint_payload:
            raise ValueError(
                f"checkpoint payload is missing required key: {required_key}"
            )

    metadata = checkpoint_payload["metadata"]

    if not isinstance(metadata, dict):
        raise ValueError("checkpoint metadata must be a dictionary.")

    required_metadata_keys = (
        "epoch",
        "project_name",
        "project_version",
    )

    for required_key in required_metadata_keys:
        if required_key not in metadata:
            raise ValueError(
                f"checkpoint metadata is missing required key: {required_key}"
            )

    saved_epoch = metadata["epoch"]

    if not isinstance(saved_epoch, int):
        raise ValueError("checkpoint metadata epoch must be an integer.")

    if saved_epoch <= 0:
        raise ValueError("checkpoint metadata epoch must be a positive integer.")


# ---------------------------------------------------------------------------
# save_checkpoint
#
# This function saves model state, optimizer state, and metadata to one
# checkpoint file.
# ---------------------------------------------------------------------------
def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    config: AppConfig,
    epoch: int,
    checkpoint_path: str | Path,
) -> Path:
    _validate_model(model)
    _validate_optimizer(optimizer)

    resolved_checkpoint_path = _validate_checkpoint_path(checkpoint_path)
    _ensure_checkpoint_parent_directory(resolved_checkpoint_path)

    metadata = _build_checkpoint_metadata(
        config=config,
        epoch=epoch,
    )
    checkpoint_payload = _build_checkpoint_payload(
        model=model,
        optimizer=optimizer,
        metadata=metadata,
    )

    torch.save(checkpoint_payload, resolved_checkpoint_path)

    return resolved_checkpoint_path


# ---------------------------------------------------------------------------
# load_checkpoint
#
# This function loads one checkpoint file and restores model and optimizer
# state. It returns the saved metadata.
# ---------------------------------------------------------------------------
def load_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    checkpoint_path: str | Path,
    map_location: str | torch.device = "cpu",
) -> CheckpointMetadata:
    _validate_model(model)
    _validate_optimizer(optimizer)

    resolved_checkpoint_path = _validate_checkpoint_path(checkpoint_path)

    if not resolved_checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint file was not found: {resolved_checkpoint_path}"
        )

    checkpoint_payload = torch.load(
        resolved_checkpoint_path,
        map_location=map_location,
        weights_only=False,
    )
    _validate_checkpoint_payload(checkpoint_payload)

    model.load_state_dict(checkpoint_payload["model_state_dict"])
    optimizer.load_state_dict(checkpoint_payload["optimizer_state_dict"])

    metadata = checkpoint_payload["metadata"]

    return CheckpointMetadata(
        epoch=metadata["epoch"],
        project_name=metadata["project_name"],
        project_version=metadata["project_version"],
    )
