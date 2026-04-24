from __future__ import annotations

import torch
from torch import nn


# ---------------------------------------------------------------------------
# validate_generation_model
#
# This function checks that the generation model is a PyTorch module.
# ---------------------------------------------------------------------------
def validate_generation_model(model: nn.Module) -> None:
    if not isinstance(model, nn.Module):
        raise TypeError("model must be an instance of torch.nn.Module.")


# ---------------------------------------------------------------------------
# validate_prompt_token_ids
#
# This function checks that prompt token IDs are a 2D long tensor.
# Expected shape: [batch_size, sequence_length]
# ---------------------------------------------------------------------------
def validate_prompt_token_ids(prompt_token_ids: torch.Tensor) -> None:
    if not isinstance(prompt_token_ids, torch.Tensor):
        raise TypeError("prompt_token_ids must be a torch.Tensor.")

    if prompt_token_ids.dim() != 2:
        raise ValueError(
            "prompt_token_ids must have shape [batch_size, sequence_length]."
        )

    if prompt_token_ids.dtype != torch.long:
        raise TypeError("prompt_token_ids must use torch.long dtype.")

    if prompt_token_ids.shape[1] <= 0:
        raise ValueError("prompt_token_ids must include at least one token.")


# ---------------------------------------------------------------------------
# validate_max_new_tokens
#
# This function checks that max_new_tokens is a positive integer.
# ---------------------------------------------------------------------------
def validate_max_new_tokens(max_new_tokens: int) -> None:
    if not isinstance(max_new_tokens, int):
        raise TypeError("max_new_tokens must be an integer.")

    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be greater than zero.")


# ---------------------------------------------------------------------------
# validate_generation_temperature
#
# This function checks that generation temperature is a positive real number.
# ---------------------------------------------------------------------------
def validate_generation_temperature(temperature: float) -> None:
    if not isinstance(temperature, (int, float)):
        raise TypeError("temperature must be a real number.")

    if temperature <= 0:
        raise ValueError("temperature must be greater than zero.")


# ---------------------------------------------------------------------------
# validate_generation_top_k
#
# This function checks that top_k is either None or a positive integer.
# ---------------------------------------------------------------------------
def validate_generation_top_k(top_k: int | None) -> None:
    if top_k is None:
        return

    if not isinstance(top_k, int):
        raise TypeError("top_k must be an integer or None.")

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")
