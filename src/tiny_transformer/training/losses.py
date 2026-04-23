from __future__ import annotations

import torch
from torch.nn import functional


# ---------------------------------------------------------------------------
# _validate_logits
#
# This function checks that logits are provided as a 3D tensor.
# Expected shape: [batch_size, sequence_length, vocab_size]
# ---------------------------------------------------------------------------
def _validate_logits(logits: torch.Tensor) -> None:
    if not isinstance(logits, torch.Tensor):
        raise TypeError("logits must be a torch.Tensor.")

    if logits.dim() != 3:
        raise ValueError(
            "logits must have shape [batch_size, sequence_length, vocab_size]."
        )


# ---------------------------------------------------------------------------
# _validate_targets
#
# This function checks that targets are provided as a 2D tensor.
# Expected shape: [batch_size, sequence_length]
# Target token IDs must use torch.long dtype.
# ---------------------------------------------------------------------------
def _validate_targets(targets: torch.Tensor) -> None:
    if not isinstance(targets, torch.Tensor):
        raise TypeError("targets must be a torch.Tensor.")

    if targets.dim() != 2:
        raise ValueError("targets must have shape [batch_size, sequence_length].")

    if targets.dtype != torch.long:
        raise TypeError("targets must use torch.long dtype.")


# ---------------------------------------------------------------------------
# _validate_matching_batch_and_sequence_shape
#
# This function checks that logits and targets agree across the batch and
# sequence dimensions.
# ---------------------------------------------------------------------------
def _validate_matching_batch_and_sequence_shape(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> None:
    logits_batch_size, logits_sequence_length, _ = logits.shape
    targets_batch_size, targets_sequence_length = targets.shape

    if logits_batch_size != targets_batch_size:
        raise ValueError("logits and targets must have the same batch_size.")

    if logits_sequence_length != targets_sequence_length:
        raise ValueError("logits and targets must have the same sequence_length.")


# ---------------------------------------------------------------------------
# _flatten_logits_for_loss
#
# This function reshapes logits so cross-entropy can score all token positions
# as one flat batch.
#
# Input shape:
# [batch_size, sequence_length, vocab_size]
#
# Output shape:
# [batch_size * sequence_length, vocab_size]
# ---------------------------------------------------------------------------
def _flatten_logits_for_loss(logits: torch.Tensor) -> torch.Tensor:
    _, _, vocab_size = logits.shape

    return logits.reshape(-1, vocab_size)


# ---------------------------------------------------------------------------
# _flatten_targets_for_loss
#
# This function reshapes targets so they line up with flattened logits.
#
# Input shape:
# [batch_size, sequence_length]
#
# Output shape:
# [batch_size * sequence_length]
# ---------------------------------------------------------------------------
def _flatten_targets_for_loss(targets: torch.Tensor) -> torch.Tensor:
    return targets.reshape(-1)


# ---------------------------------------------------------------------------
# compute_next_token_loss
#
# This function computes cross-entropy loss for next-token prediction.
# The model outputs one vocab score vector for each token position.
# The target tensor stores the correct next token ID at each position.
# ---------------------------------------------------------------------------
def compute_next_token_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    _validate_logits(logits)
    _validate_targets(targets)
    _validate_matching_batch_and_sequence_shape(logits, targets)

    flattened_logits = _flatten_logits_for_loss(logits)
    flattened_targets = _flatten_targets_for_loss(targets)

    return functional.cross_entropy(flattened_logits, flattened_targets)
