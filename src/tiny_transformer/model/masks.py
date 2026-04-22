from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# _validate_positive_int
#
# This function checks that a value is a positive integer.
# ---------------------------------------------------------------------------
def _validate_positive_int(value: int, field_name: str) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer.")


# ---------------------------------------------------------------------------
# create_causal_attention_mask
#
# This function creates a causal mask for decoder self-attention.
# A True value means the position is blocked.
#
# Output shape:
# [sequence_length, sequence_length]
# ---------------------------------------------------------------------------
def create_causal_attention_mask(sequence_length: int) -> torch.Tensor:
    _validate_positive_int(sequence_length, "sequence_length")

    # The upper triangle above the main diagonal marks future positions.
    # Those positions must be blocked in decoder self-attention.
    mask = torch.triu(
        torch.ones(sequence_length, sequence_length, dtype=torch.bool),
        diagonal=1,
    )

    return mask
