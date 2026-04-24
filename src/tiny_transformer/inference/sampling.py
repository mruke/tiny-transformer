from __future__ import annotations

import torch


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
# greedy_sample_next_token
#
# This function selects the highest-logit token from each batch row.
# It returns one token ID per batch row.
# ---------------------------------------------------------------------------
def greedy_sample_next_token(logits: torch.Tensor) -> torch.Tensor:
    _validate_logits(logits)

    return torch.argmax(logits, dim=1)
