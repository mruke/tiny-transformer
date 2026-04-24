from __future__ import annotations

import torch
from torch.nn import functional

from tiny_transformer.inference.sampling.top_k import apply_top_k_filter


# ---------------------------------------------------------------------------
# _validate_logits
#
# This function checks that logits are a 2D tensor.
# Expected shape: [batch_size, vocab_size]
# ---------------------------------------------------------------------------
def _validate_logits(logits: torch.Tensor) -> None:
    if not isinstance(logits, torch.Tensor):
        raise TypeError("logits must be a torch.Tensor.")

    if logits.dim() != 2:
        raise ValueError("logits must have shape [batch_size, vocab_size].")

    if logits.shape[1] <= 0:
        raise ValueError("logits must include a non-empty vocab_size dimension.")


# ---------------------------------------------------------------------------
# _validate_temperature
#
# This function checks that temperature is a positive real number.
# ---------------------------------------------------------------------------
def _validate_temperature(temperature: float) -> None:
    if not isinstance(temperature, (int, float)):
        raise TypeError("temperature must be a real number.")

    if temperature <= 0:
        raise ValueError("temperature must be greater than zero.")


# ---------------------------------------------------------------------------
# _scale_logits_by_temperature
#
# This function scales logits by temperature before sampling.
# Lower temperatures make the distribution sharper.
# Higher temperatures make the distribution flatter.
# ---------------------------------------------------------------------------
def _scale_logits_by_temperature(
    logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    return logits / float(temperature)


# ---------------------------------------------------------------------------
# temperature_sample_next_token
#
# This function samples one token from each batch row after scaling logits by
# temperature and converting them to probabilities.
# ---------------------------------------------------------------------------
def temperature_sample_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: int | None = None,
) -> torch.Tensor:
    _validate_logits(logits)
    _validate_temperature(temperature)

    scaled_logits = _scale_logits_by_temperature(
        logits=logits,
        temperature=temperature,
    )
    filtered_logits = apply_top_k_filter(
        logits=scaled_logits,
        top_k=top_k,
    )
    probabilities = functional.softmax(filtered_logits, dim=1)
    sampled_token_ids = torch.multinomial(probabilities, num_samples=1)

    return sampled_token_ids.squeeze(1)
