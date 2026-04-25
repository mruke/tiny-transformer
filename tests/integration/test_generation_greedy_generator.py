from __future__ import annotations

import torch

from tests.helpers.generation_test_models import (
    ContextWindowTrackingModel,
    GreedyTestModel,
)
from tiny_transformer.inference.generator import generate_next_tokens_greedy


# ---------------------------------------------------------------------------
# test_generate_next_tokens_greedy_appends_expected_number_of_tokens
#
# This test checks that generation adds the requested number of new tokens.
# ---------------------------------------------------------------------------
def test_generate_next_tokens_greedy_appends_expected_number_of_tokens() -> None:
    model = GreedyTestModel(vocab_size=8, next_token_id=3)
    prompt_token_ids = torch.tensor([[1, 2, 4]], dtype=torch.long)

    generated_token_ids = generate_next_tokens_greedy(
        model=model,
        prompt_token_ids=prompt_token_ids,
        max_new_tokens=2,
    )

    assert generated_token_ids.shape == (1, 5)


# ---------------------------------------------------------------------------
# test_generate_next_tokens_greedy_preserves_prompt_prefix
#
# This test checks that generated output keeps the original prompt tokens.
# ---------------------------------------------------------------------------
def test_generate_next_tokens_greedy_preserves_prompt_prefix() -> None:
    model = GreedyTestModel(vocab_size=8, next_token_id=3)
    prompt_token_ids = torch.tensor([[1, 2, 4]], dtype=torch.long)

    generated_token_ids = generate_next_tokens_greedy(
        model=model,
        prompt_token_ids=prompt_token_ids,
        max_new_tokens=2,
    )

    assert torch.equal(generated_token_ids[:, :3], prompt_token_ids)


# ---------------------------------------------------------------------------
# test_generate_next_tokens_greedy_appends_expected_token_ids
#
# This test checks that greedy generation appends the model's highest-logit
# token choice.
# ---------------------------------------------------------------------------
def test_generate_next_tokens_greedy_appends_expected_token_ids() -> None:
    model = GreedyTestModel(vocab_size=8, next_token_id=3)
    prompt_token_ids = torch.tensor([[1, 2, 4]], dtype=torch.long)

    generated_token_ids = generate_next_tokens_greedy(
        model=model,
        prompt_token_ids=prompt_token_ids,
        max_new_tokens=3,
    )

    expected_token_ids = torch.tensor([[1, 2, 4, 3, 3, 3]], dtype=torch.long)

    assert torch.equal(generated_token_ids, expected_token_ids)


# ---------------------------------------------------------------------------
# test_generate_next_tokens_greedy_supports_batch_generation
#
# This test checks that generation works for more than one batch row.
# ---------------------------------------------------------------------------
def test_generate_next_tokens_greedy_supports_batch_generation() -> None:
    model = GreedyTestModel(vocab_size=8, next_token_id=5)
    prompt_token_ids = torch.tensor(
        [
            [1, 2, 4],
            [3, 0, 6],
        ],
        dtype=torch.long,
    )

    generated_token_ids = generate_next_tokens_greedy(
        model=model,
        prompt_token_ids=prompt_token_ids,
        max_new_tokens=2,
    )

    assert generated_token_ids.shape == (2, 5)
    assert torch.equal(generated_token_ids[:, -2:], torch.tensor([[5, 5], [5, 5]]))


# ---------------------------------------------------------------------------
# test_generate_next_tokens_greedy_trims_context_to_model_window
#
# This test checks that generation trims the running context to the model's
# max sequence length before each forward pass.
# ---------------------------------------------------------------------------
def test_generate_next_tokens_greedy_trims_context_to_model_window() -> None:
    model = ContextWindowTrackingModel(
        vocab_size=8,
        next_token_id=3,
        max_sequence_length=3,
    )
    prompt_token_ids = torch.tensor([[1, 2, 4]], dtype=torch.long)

    generated_token_ids = generate_next_tokens_greedy(
        model=model,
        prompt_token_ids=prompt_token_ids,
        max_new_tokens=2,
    )

    assert generated_token_ids.shape == (1, 5)
    assert model.seen_sequence_lengths == [3, 3]
