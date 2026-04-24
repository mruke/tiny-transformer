from __future__ import annotations

import torch
from torch import nn


# ---------------------------------------------------------------------------
# validate_generation_model
#
# This function will validate the model passed into the generator flow.
# ---------------------------------------------------------------------------
def validate_generation_model(model: nn.Module) -> None:
    raise NotImplementedError("Generation validation will be implemented in Commit 2.")


# ---------------------------------------------------------------------------
# validate_prompt_token_ids
#
# This function will validate prompt token IDs for generation.
# ---------------------------------------------------------------------------
def validate_prompt_token_ids(prompt_token_ids: torch.Tensor) -> None:
    raise NotImplementedError("Generation validation will be implemented in Commit 2.")


# ---------------------------------------------------------------------------
# validate_max_new_tokens
#
# This function will validate the requested number of new tokens.
# ---------------------------------------------------------------------------
def validate_max_new_tokens(max_new_tokens: int) -> None:
    raise NotImplementedError("Generation validation will be implemented in Commit 2.")
