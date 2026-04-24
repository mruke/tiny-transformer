from __future__ import annotations

import pytest
import torch
from torch import nn

from tiny_transformer.inference.generator import (
    generate_next_tokens,
    generate_next_tokens_greedy,
)


# ---------------------------------------------------------------------------
# _GreedyTestModel
#
# This test model returns fixed logits so generation behavior stays
# deterministic.
# ---------------------------------------------------------------------------
class _GreedyTestModel(nn.Module):
    # -----------------------------------------------------------------------
    # _GreedyTestModel.__init__
    #
    # This method stores the vocab size and fixed next-token choice.
    # -----------------------------------------------------------------------
    def __init__(
        self,
        vocab_size: int,
        next_token_id: int,
    ) -> None:
        super().__init__()
        self._vocab_size = vocab_size
        self._next_token_id = next_token_id

    # -----------------------------------------------------------------------
    # _GreedyTestModel.forward
    #
    # This method returns logits that always favor the same next token.
    # -----------------------------------------------------------------------
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length = input_ids.shape
        logits = torch.zeros(
            batch_size,
            sequence_length,
            self._vocab_size,
            dtype=torch.float32,
            device=input_ids.device,
        )
        logits[:, :, self._next_token_id] = 1.0

        return logits


# ---------------------------------------------------------------------------
# _TemperatureTestModel
#
# This test model returns fixed logits with two strong token choices so
# temperature sampling and top-k behavior can be exercised predictably.
# ---------------------------------------------------------------------------
class _TemperatureTestModel(nn.Module):
    # -----------------------------------------------------------------------
    # _TemperatureTestModel.__init__
    #
    # This method stores the vocab size.
    # -----------------------------------------------------------------------
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self._vocab_size = vocab_size

    # -----------------------------------------------------------------------
    # _TemperatureTestModel.forward
    #
    # This method returns logits with token 1 and token 2 as the strongest
    # choices at every step.
    # -----------------------------------------------------------------------
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length = input_ids.shape
        logits = torch.full(
            (batch_size, sequence_length, self._vocab_size),
            fill_value=-10.0,
            dtype=torch.float32,
            device=input_ids.device,
        )
        logits[:, :, 1] = 10.0
        logits[:, :, 2] = 9.0

        return logits


# ---------------------------------------------------------------------------
# greedy generation tests
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# test_generate_next_tokens_greedy_appends_expected_number_of_tokens
#
# This test checks that generation adds the requested number of new tokens.
# ---------------------------------------------------------------------------
def test_generate_next_tokens_greedy_appends_expected_number_of_tokens() -> None:
    model = _GreedyTestModel(vocab_size=8, next_token_id=3)
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
    model = _GreedyTestModel(vocab_size=8, next_token_id=3)
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
    model = _GreedyTestModel(vocab_size=8, next_token_id=3)
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
    model = _GreedyTestModel(vocab_size=8, next_token_id=5)
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
# temperature generation tests
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# test_generate_next_tokens_returns_expected_shape
#
# This test checks that temperature-based generation appends the requested
# number of new tokens.
# ---------------------------------------------------------------------------
def test_generate_next_tokens_returns_expected_shape() -> None:
    torch.manual_seed(0)
    model = _TemperatureTestModel(vocab_size=6)
    prompt_token_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

    generated_token_ids = generate_next_tokens(
        model=model,
        prompt_token_ids=prompt_token_ids,
        max_new_tokens=2,
        temperature=1.0,
    )

    assert generated_token_ids.shape == (1, 5)


# ---------------------------------------------------------------------------
# test_generate_next_tokens_preserves_prompt_prefix
#
# This test checks that temperature-based generation keeps the original
# prompt tokens.
# ---------------------------------------------------------------------------
def test_generate_next_tokens_preserves_prompt_prefix() -> None:
    torch.manual_seed(0)
    model = _TemperatureTestModel(vocab_size=6)
    prompt_token_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

    generated_token_ids = generate_next_tokens(
        model=model,
        prompt_token_ids=prompt_token_ids,
        max_new_tokens=2,
        temperature=1.0,
    )

    assert torch.equal(generated_token_ids[:, :3], prompt_token_ids)


# ---------------------------------------------------------------------------
# test_generate_next_tokens_with_top_k_limits_generated_choices
#
# This test checks that top_k=2 limits sampled tokens to the top two model
# choices.
# ---------------------------------------------------------------------------
def test_generate_next_tokens_with_top_k_limits_generated_choices() -> None:
    torch.manual_seed(0)
    model = _TemperatureTestModel(vocab_size=6)
    prompt_token_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

    generated_token_ids = generate_next_tokens(
        model=model,
        prompt_token_ids=prompt_token_ids,
        max_new_tokens=4,
        temperature=1.0,
        top_k=2,
    )

    appended_token_ids = generated_token_ids[:, 3:]

    assert torch.all((appended_token_ids == 1) | (appended_token_ids == 2))


# ---------------------------------------------------------------------------
# test_generate_next_tokens_rejects_non_numeric_temperature
#
# This test checks that temperature must be numeric.
# ---------------------------------------------------------------------------
def test_generate_next_tokens_rejects_non_numeric_temperature() -> None:
    model = _TemperatureTestModel(vocab_size=6)
    prompt_token_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

    with pytest.raises(TypeError, match="temperature must be a real number"):
        generate_next_tokens(
            model=model,
            prompt_token_ids=prompt_token_ids,
            max_new_tokens=2,
            temperature="1.0",  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# test_generate_next_tokens_rejects_non_positive_temperature
#
# This test checks that temperature must be greater than zero.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("temperature", [0.0, -1.0])
def test_generate_next_tokens_rejects_non_positive_temperature(
    temperature: float,
) -> None:
    model = _TemperatureTestModel(vocab_size=6)
    prompt_token_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

    with pytest.raises(ValueError, match="temperature must be greater than zero"):
        generate_next_tokens(
            model=model,
            prompt_token_ids=prompt_token_ids,
            max_new_tokens=2,
            temperature=temperature,
        )


# ---------------------------------------------------------------------------
# test_generate_next_tokens_rejects_non_integer_top_k
#
# This test checks that top_k must be an integer or None.
# ---------------------------------------------------------------------------
def test_generate_next_tokens_rejects_non_integer_top_k() -> None:
    model = _TemperatureTestModel(vocab_size=6)
    prompt_token_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

    with pytest.raises(TypeError, match="top_k must be an integer or None"):
        generate_next_tokens(
            model=model,
            prompt_token_ids=prompt_token_ids,
            max_new_tokens=2,
            temperature=1.0,
            top_k=2.5,  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# test_generate_next_tokens_rejects_non_positive_top_k
#
# This test checks that top_k must be greater than zero.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("top_k", [0, -1])
def test_generate_next_tokens_rejects_non_positive_top_k(top_k: int) -> None:
    model = _TemperatureTestModel(vocab_size=6)
    prompt_token_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

    with pytest.raises(ValueError, match="top_k must be greater than zero"):
        generate_next_tokens(
            model=model,
            prompt_token_ids=prompt_token_ids,
            max_new_tokens=2,
            temperature=1.0,
            top_k=top_k,
        )
