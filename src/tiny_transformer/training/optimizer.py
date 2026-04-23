from __future__ import annotations

from typing import Iterator

from torch import nn
from torch.optim import AdamW, Optimizer

from tiny_transformer.config import AppConfig


# ---------------------------------------------------------------------------
# _validate_model
#
# This function checks that the model is a PyTorch module.
# ---------------------------------------------------------------------------
def _validate_model(model: nn.Module) -> None:
    if not isinstance(model, nn.Module):
        raise TypeError("model must be an instance of torch.nn.Module.")


# ---------------------------------------------------------------------------
# _validate_has_parameters
#
# This function checks that the model has parameters to optimize.
# ---------------------------------------------------------------------------
def _validate_has_parameters(parameters: list[nn.Parameter]) -> None:
    if not parameters:
        raise ValueError("model must have at least one parameter to optimize.")


# ---------------------------------------------------------------------------
# _get_trainable_parameters
#
# This function collects model parameters that require gradients.
# Frozen parameters are skipped because the optimizer should not update them.
# ---------------------------------------------------------------------------
def _get_trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    parameters: Iterator[nn.Parameter] = model.parameters()

    return [parameter for parameter in parameters if parameter.requires_grad]


# ---------------------------------------------------------------------------
# build_optimizer
#
# This function builds the training optimizer from app config values.
# The current training setup uses AdamW with learning rate and weight decay
# from the training config section.
# ---------------------------------------------------------------------------
def build_optimizer(
    model: nn.Module,
    config: AppConfig,
) -> Optimizer:
    _validate_model(model)

    trainable_parameters = _get_trainable_parameters(model)
    _validate_has_parameters(trainable_parameters)

    return AdamW(
        trainable_parameters,
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
