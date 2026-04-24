from __future__ import annotations

import torch
from torch import nn


# ---------------------------------------------------------------------------
# generate_next_tokens_greedy
#
# This function will generate new token IDs autoregressively using greedy
# sampling.
# ---------------------------------------------------------------------------
def generate_next_tokens_greedy(
    model: nn.Module,
    prompt_token_ids: torch.Tensor,
    max_new_tokens: int,
) -> torch.Tensor:
    raise NotImplementedError("Greedy generation will be implemented in Commit 2.")
