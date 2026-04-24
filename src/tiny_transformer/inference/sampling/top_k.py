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
# _validate_top_k
#
# This function checks that top_k is either None or a positive integer that
# does not exceed vocab size.
# ---------------------------------------------------------------------------
def _validate_top_k(
    top_k: int | None,
    vocab_size: int,
) -> None:
    if top_k is None:
        return

    if not isinstance(top_k, int):
        raise TypeError("top_k must be an integer or None.")

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    if top_k > vocab_size:
        raise ValueError("top_k must not exceed vocab_size.")


# ---------------------------------------------------------------------------
# apply_top_k_filter
#
# This function keeps only the top-k logits in each batch row.
# All other logits are replaced with negative infinity so they cannot be
# sampled after softmax.
# ---------------------------------------------------------------------------
def apply_top_k_filter(
    logits: torch.Tensor,
    top_k: int | None,
) -> torch.Tensor:
    _validate_logits(logits)
    _validate_top_k(top_k, vocab_size=logits.shape[1])

    if top_k is None:
        return logits

    filtered_logits = logits.clone()
    top_k_values, _ = torch.topk(filtered_logits, k=top_k, dim=1)
    minimum_kept_values = top_k_values[:, -1].unsqueeze(1)

    filtered_logits[filtered_logits < minimum_kept_values] = float("-inf")

    return filtered_logits
