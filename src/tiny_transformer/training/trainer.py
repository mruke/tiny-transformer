from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn
from torch.optim import Optimizer

from tiny_transformer.training.losses import compute_next_token_loss
from tiny_transformer.training.metrics import LossTracker


# ---------------------------------------------------------------------------
# TrainEpochResult
#
# TrainEpochResult stores the summary values from one training epoch.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TrainEpochResult:
    average_loss: float
    batch_count: int


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
# _validate_device
#
# This function checks that the device is a non-empty string.
# ---------------------------------------------------------------------------
def _validate_device(device: str) -> None:
    if not isinstance(device, str) or not device.strip():
        raise ValueError("device must be a non-empty string.")


# ---------------------------------------------------------------------------
# _validate_token_batch
#
# This function checks that one token batch is a 2D long tensor.
# Expected shape: [batch_size, sequence_length]
# ---------------------------------------------------------------------------
def _validate_token_batch(batch: torch.Tensor, batch_name: str) -> None:
    if not isinstance(batch, torch.Tensor):
        raise TypeError(f"{batch_name} must be a torch.Tensor.")

    if batch.dim() != 2:
        raise ValueError(f"{batch_name} must have shape [batch_size, sequence_length].")

    if batch.dtype != torch.long:
        raise TypeError(f"{batch_name} must use torch.long dtype.")


# ---------------------------------------------------------------------------
# _validate_matching_batch_shape
#
# This function checks that input IDs and target IDs line up.
# ---------------------------------------------------------------------------
def _validate_matching_batch_shape(
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
) -> None:
    if input_ids.shape != target_ids.shape:
        raise ValueError("input_ids and target_ids must have the same shape.")


# ---------------------------------------------------------------------------
# _validate_batch_count
#
# This function checks that at least one batch was processed.
# ---------------------------------------------------------------------------
def _validate_batch_count(batch_count: int) -> None:
    if batch_count <= 0:
        raise ValueError("Training epoch must process at least one batch.")


# ---------------------------------------------------------------------------
# Trainer
#
# Trainer runs training batches and training epochs for the language model.
# This class keeps training orchestration separate from:
# - model definition
# - loss calculation
# - optimizer construction
# ---------------------------------------------------------------------------
class Trainer:
    # -----------------------------------------------------------------------
    # Trainer.__init__
    #
    # This method stores the model, optimizer, and target device.
    # The model is moved to the target device during setup.
    # -----------------------------------------------------------------------
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        device: str,
    ) -> None:
        _validate_model(model)
        _validate_optimizer(optimizer)
        _validate_device(device)

        self._model = model.to(device)
        self._optimizer = optimizer
        self._device = device

    # -----------------------------------------------------------------------
    # device
    #
    # This property returns the trainer device string.
    # -----------------------------------------------------------------------
    @property
    def device(self) -> str:
        return self._device

    # -----------------------------------------------------------------------
    # _prepare_batch
    #
    # This method validates one training batch and moves it to the trainer
    # device.
    # -----------------------------------------------------------------------
    def _prepare_batch(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _validate_token_batch(input_ids, "input_ids")
        _validate_token_batch(target_ids, "target_ids")
        _validate_matching_batch_shape(input_ids, target_ids)

        return input_ids.to(self._device), target_ids.to(self._device)

    # -----------------------------------------------------------------------
    # train_batch
    #
    # This method runs one optimizer update on one batch.
    # It returns the scalar loss value for that batch.
    # -----------------------------------------------------------------------
    def train_batch(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> float:
        prepared_input_ids, prepared_target_ids = self._prepare_batch(
            input_ids=input_ids,
            target_ids=target_ids,
        )

        self._model.train()
        self._optimizer.zero_grad(set_to_none=True)

        logits = self._model(prepared_input_ids)
        loss = compute_next_token_loss(logits, prepared_target_ids)

        loss.backward()
        self._optimizer.step()

        return float(loss.detach().item())

    # -----------------------------------------------------------------------
    # train_epoch
    #
    # This method runs one full pass across all training batches.
    # It returns the average loss and batch count for the epoch.
    # -----------------------------------------------------------------------
    def train_epoch(
        self,
        batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    ) -> TrainEpochResult:
        total_loss = 0.0
        batch_count = 0

        for input_ids, target_ids in batches:
            batch_loss = self.train_batch(
                input_ids=input_ids,
                target_ids=target_ids,
            )
            total_loss += batch_loss
            batch_count += 1

        _validate_batch_count(batch_count)

        average_loss = total_loss / batch_count

        return TrainEpochResult(
            average_loss=average_loss,
            batch_count=batch_count,
        )

    # -----------------------------------------------------------------------
    # validate_batch
    #
    # This method computes loss for one validation batch.
    # No gradients or optimizer updates are used in validation mode.
    # -----------------------------------------------------------------------
    def validate_batch(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> float:
        prepared_input_ids, prepared_target_ids = self._prepare_batch(
            input_ids=input_ids,
            target_ids=target_ids,
        )

        self._model.eval()

        with torch.no_grad():
            logits = self._model(prepared_input_ids)
            loss = compute_next_token_loss(logits, prepared_target_ids)

        return float(loss.item())

    # -----------------------------------------------------------------------
    # validate_epoch
    #
    # This method runs one full pass across all validation batches.
    # It returns the average loss and batch count for the epoch.
    # -----------------------------------------------------------------------
    def validate_epoch(
        self,
        batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    ) -> TrainEpochResult:
        loss_tracker = LossTracker()

        for input_ids, target_ids in batches:
            batch_loss = self.validate_batch(
                input_ids=input_ids,
                target_ids=target_ids,
            )
            loss_tracker.update(batch_loss)

        _validate_batch_count(loss_tracker.batch_count)

        return TrainEpochResult(
            average_loss=loss_tracker.average_loss(),
            batch_count=loss_tracker.batch_count,
        )
