from __future__ import annotations

import torch
from torch import nn

from tiny_transformer.inference.generation_validation import (
    validate_generation_model,
    validate_generation_temperature,
    validate_generation_top_k,
    validate_max_new_tokens,
    validate_prompt_token_ids,
)
from tiny_transformer.inference.sampling.greedy import greedy_sample_next_token
from tiny_transformer.inference.sampling.temperature import (
    temperature_sample_next_token,
)


# ---------------------------------------------------------------------------
# _append_next_token_ids
#
# This function appends one new token ID column to the generated sequence.
# ---------------------------------------------------------------------------
def _append_next_token_ids(
    generated_token_ids: torch.Tensor,
    next_token_ids: torch.Tensor,
) -> torch.Tensor:
    return torch.cat((generated_token_ids, next_token_ids), dim=1)


# ---------------------------------------------------------------------------
# _extract_next_token_logits
#
# This function returns the logits for the final sequence position in each
# batch row.
# ---------------------------------------------------------------------------
def _extract_next_token_logits(logits: torch.Tensor) -> torch.Tensor:
    return logits[:, -1, :]


# ---------------------------------------------------------------------------
# _resolve_model_device
#
# This function returns the model device when parameters or buffers exist.
# Otherwise, it falls back to the prompt tensor device.
# ---------------------------------------------------------------------------
def _resolve_model_device(
    model: nn.Module,
    prompt_token_ids: torch.Tensor,
) -> torch.device:
    first_parameter = next(model.parameters(), None)

    if first_parameter is not None:
        return first_parameter.device

    first_buffer = next(model.buffers(), None)

    if first_buffer is not None:
        return first_buffer.device

    return prompt_token_ids.device


# ---------------------------------------------------------------------------
# _resolve_model_context_window
#
# This function returns the model max sequence length when the model exposes
# a valid max_sequence_length property. Otherwise, it returns None.
# ---------------------------------------------------------------------------
def _resolve_model_context_window(model: nn.Module) -> int | None:
    max_sequence_length = getattr(model, "max_sequence_length", None)

    if max_sequence_length is None:
        return None

    if not isinstance(max_sequence_length, int) or max_sequence_length <= 0:
        raise ValueError("model max_sequence_length must be a positive integer.")

    return max_sequence_length


# ---------------------------------------------------------------------------
# _trim_generation_context
#
# This function trims generated token IDs to the model context window when one
# is available.
# ---------------------------------------------------------------------------
def _trim_generation_context(
    generated_token_ids: torch.Tensor,
    context_window: int | None,
) -> torch.Tensor:
    if context_window is None:
        return generated_token_ids

    _, sequence_length = generated_token_ids.shape

    if sequence_length <= context_window:
        return generated_token_ids

    return generated_token_ids[:, -context_window:]


# ---------------------------------------------------------------------------
# _run_generation_loop
#
# This function runs the shared autoregressive token generation loop.
# The provided sampler function chooses the next token from the final-step
# logits for each iteration.
# ---------------------------------------------------------------------------
def _run_generation_loop(
    model: nn.Module,
    prompt_token_ids: torch.Tensor,
    max_new_tokens: int,
    sampler: callable,
) -> torch.Tensor:
    model_device = _resolve_model_device(model, prompt_token_ids)
    context_window = _resolve_model_context_window(model)
    generated_token_ids = prompt_token_ids.to(model_device).clone()

    model.eval()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            model_input_ids = _trim_generation_context(
                generated_token_ids=generated_token_ids,
                context_window=context_window,
            )
            logits = model(model_input_ids)
            next_token_logits = _extract_next_token_logits(logits)
            next_token_ids = sampler(next_token_logits).unsqueeze(1)

            generated_token_ids = _append_next_token_ids(
                generated_token_ids=generated_token_ids,
                next_token_ids=next_token_ids,
            )

    return generated_token_ids


# ---------------------------------------------------------------------------
# generate_next_tokens_greedy
#
# This function generates new token IDs autoregressively using greedy
# next-token selection.
# ---------------------------------------------------------------------------
def generate_next_tokens_greedy(
    model: nn.Module,
    prompt_token_ids: torch.Tensor,
    max_new_tokens: int,
) -> torch.Tensor:
    validate_generation_model(model)
    validate_prompt_token_ids(prompt_token_ids)
    validate_max_new_tokens(max_new_tokens)

    return _run_generation_loop(
        model=model,
        prompt_token_ids=prompt_token_ids,
        max_new_tokens=max_new_tokens,
        sampler=greedy_sample_next_token,
    )


# ---------------------------------------------------------------------------
# generate_next_tokens
#
# This function generates new token IDs autoregressively using temperature
# sampling with optional top-k filtering.
# ---------------------------------------------------------------------------
def generate_next_tokens(
    model: nn.Module,
    prompt_token_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None = None,
) -> torch.Tensor:
    validate_generation_model(model)
    validate_prompt_token_ids(prompt_token_ids)
    validate_max_new_tokens(max_new_tokens)
    validate_generation_temperature(temperature)
    validate_generation_top_k(top_k)

    def sample_next_token(next_token_logits: torch.Tensor) -> torch.Tensor:
        return temperature_sample_next_token(
            logits=next_token_logits,
            temperature=temperature,
            top_k=top_k,
        )

    return _run_generation_loop(
        model=model,
        prompt_token_ids=prompt_token_ids,
        max_new_tokens=max_new_tokens,
        sampler=sample_next_token,
    )
