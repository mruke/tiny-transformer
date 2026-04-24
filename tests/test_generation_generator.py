from __future__ import annotations

import pytest
import torch
from torch import nn

from tiny_transformer.inference.generator import generate_next_tokens_greedy


# ---------------------------------------------------------------------------
# _GeneratorScaffoldTestModel
#
# This test model exists only to confirm the generator scaffold import path.
# ---------------------------------------------------------------------------
class _GeneratorScaffoldTestModel(nn.Module):
    # -----------------------------------------------------------------------
    # _GeneratorScaffoldTestModel.forward
    #
    # This method returns placeholder logits for scaffold testing.
    # -----------------------------------------------------------------------
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length = input_ids.shape

        return torch.zeros(batch_size, sequence_length, 4, dtype=torch.float32)


# ---------------------------------------------------------------------------
# test_generate_next_tokens_greedy_scaffold_raises_not_implemented
#
# This test checks that the scaffolded generator clearly shows that real
# behavior has not been implemented yet.
# ---------------------------------------------------------------------------
def test_generate_next_tokens_greedy_scaffold_raises_not_implemented() -> None:
    model = _GeneratorScaffoldTestModel()
    prompt_token_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

    with pytest.raises(
        NotImplementedError, match="Greedy generation will be implemented"
    ):
        generate_next_tokens_greedy(
            model=model,
            prompt_token_ids=prompt_token_ids,
            max_new_tokens=2,
        )
