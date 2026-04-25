from __future__ import annotations

import pytest
import torch

from tests.helpers.generation_test_models import ValidationTestModel
from tiny_transformer.inference.generator import (
    generate_next_tokens,
    generate_next_tokens_greedy,
)


# ---------------------------------------------------------------------------
# test_generate_next_tokens_greedy_rejects_non_module_model
#
# This test checks that model must be a PyTorch module.
# ---------------------------------------------------------------------------
def test_generate_next_tokens_greedy_rejects_non_module_model() -> None:
    prompt_token_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

    with pytest.raises(
        TypeError,
        match="model must be an instance of torch.nn.Module",
    ):
        generate_next_tokens_greedy(
            model=object(),  # type: ignore[arg-type]
            prompt_token_ids=prompt_token_ids,
            max_new_tokens=2,
        )


# ---------------------------------------------------------------------------
# test_generate_next_tokens_greedy_rejects_non_tensor_prompt_token_ids
#
# This test checks that prompt token IDs must be a tensor.
# ---------------------------------------------------------------------------
def test_generate_next_tokens_greedy_rejects_non_tensor_prompt_token_ids() -> None:
    model = ValidationTestModel()

    with pytest.raises(TypeError, match="prompt_token_ids must be a torch.Tensor"):
        generate_next_tokens_greedy(
            model=model,
            prompt_token_ids=[[1, 2, 3]],  # type: ignore[arg-type]
            max_new_tokens=2,
        )


# ---------------------------------------------------------------------------
# test_generate_next_tokens_greedy_rejects_non_2d_prompt_token_ids
#
# This test checks that prompt token IDs must keep batch and sequence
# dimensions.
# ---------------------------------------------------------------------------
def test_generate_next_tokens_greedy_rejects_non_2d_prompt_token_ids() -> None:
    model = ValidationTestModel()
    prompt_token_ids = torch.tensor([1, 2, 3], dtype=torch.long)

    with pytest.raises(
        ValueError,
        match="prompt_token_ids must have shape \\[batch_size, sequence_length\\]",
    ):
        generate_next_tokens_greedy(
            model=model,
            prompt_token_ids=prompt_token_ids,
            max_new_tokens=2,
        )


# ---------------------------------------------------------------------------
# test_generate_next_tokens_greedy_rejects_non_long_prompt_token_ids
#
# This test checks that prompt token IDs must use torch.long dtype.
# ---------------------------------------------------------------------------
def test_generate_next_tokens_greedy_rejects_non_long_prompt_token_ids() -> None:
    model = ValidationTestModel()
    prompt_token_ids = torch.tensor([[1, 2, 3]], dtype=torch.int32)

    with pytest.raises(TypeError, match="prompt_token_ids must use torch.long dtype"):
        generate_next_tokens_greedy(
            model=model,
            prompt_token_ids=prompt_token_ids,
            max_new_tokens=2,
        )


# ---------------------------------------------------------------------------
# test_generate_next_tokens_greedy_rejects_empty_prompt_sequence
#
# This test checks that prompt token IDs must include at least one token.
# ---------------------------------------------------------------------------
def test_generate_next_tokens_greedy_rejects_empty_prompt_sequence() -> None:
    model = ValidationTestModel()
    prompt_token_ids = torch.empty((1, 0), dtype=torch.long)

    with pytest.raises(
        ValueError, match="prompt_token_ids must include at least one token"
    ):
        generate_next_tokens_greedy(
            model=model,
            prompt_token_ids=prompt_token_ids,
            max_new_tokens=2,
        )


# ---------------------------------------------------------------------------
# test_generate_next_tokens_greedy_rejects_empty_prompt_batch
#
# This test checks that prompt token IDs must include at least one batch row.
# ---------------------------------------------------------------------------
def test_generate_next_tokens_greedy_rejects_empty_prompt_batch() -> None:
    model = ValidationTestModel()
    prompt_token_ids = torch.empty((0, 3), dtype=torch.long)

    with pytest.raises(
        ValueError,
        match="prompt_token_ids must include at least one batch row",
    ):
        generate_next_tokens_greedy(
            model=model,
            prompt_token_ids=prompt_token_ids,
            max_new_tokens=2,
        )


# ---------------------------------------------------------------------------
# test_generate_next_tokens_greedy_rejects_non_integer_max_new_tokens
#
# This test checks that max_new_tokens must be an integer.
# ---------------------------------------------------------------------------
def test_generate_next_tokens_greedy_rejects_non_integer_max_new_tokens() -> None:
    model = ValidationTestModel()
    prompt_token_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

    with pytest.raises(TypeError, match="max_new_tokens must be an integer"):
        generate_next_tokens_greedy(
            model=model,
            prompt_token_ids=prompt_token_ids,
            max_new_tokens=2.5,  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# test_generate_next_tokens_greedy_rejects_non_positive_max_new_tokens
#
# This test checks that max_new_tokens must be greater than zero.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("max_new_tokens", [0, -1])
def test_generate_next_tokens_greedy_rejects_non_positive_max_new_tokens(
    max_new_tokens: int,
) -> None:
    model = ValidationTestModel()
    prompt_token_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

    with pytest.raises(ValueError, match="max_new_tokens must be greater than zero"):
        generate_next_tokens_greedy(
            model=model,
            prompt_token_ids=prompt_token_ids,
            max_new_tokens=max_new_tokens,
        )


# ---------------------------------------------------------------------------
# test_generate_next_tokens_rejects_non_module_model
#
# This test checks that model must be a PyTorch module for temperature-based
# generation too.
# ---------------------------------------------------------------------------
def test_generate_next_tokens_rejects_non_module_model() -> None:
    prompt_token_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

    with pytest.raises(
        TypeError,
        match="model must be an instance of torch.nn.Module",
    ):
        generate_next_tokens(
            model=object(),  # type: ignore[arg-type]
            prompt_token_ids=prompt_token_ids,
            max_new_tokens=2,
            temperature=1.0,
        )
