from __future__ import annotations

import pytest
import torch

from tests.helpers.generation_test_models import TemperatureTestModel
from tiny_transformer.inference.generator import generate_next_tokens


# ---------------------------------------------------------------------------
# test_generate_next_tokens_returns_expected_shape
#
# This test checks that temperature-based generation appends the requested
# number of new tokens.
# ---------------------------------------------------------------------------
def test_generate_next_tokens_returns_expected_shape() -> None:
    torch.manual_seed(0)
    model = TemperatureTestModel(vocab_size=6)
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
    model = TemperatureTestModel(vocab_size=6)
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
    model = TemperatureTestModel(vocab_size=6)
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
    model = TemperatureTestModel(vocab_size=6)
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
    model = TemperatureTestModel(vocab_size=6)
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
    model = TemperatureTestModel(vocab_size=6)
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
    model = TemperatureTestModel(vocab_size=6)
    prompt_token_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

    with pytest.raises(ValueError, match="top_k must be greater than zero"):
        generate_next_tokens(
            model=model,
            prompt_token_ids=prompt_token_ids,
            max_new_tokens=2,
            temperature=1.0,
            top_k=top_k,
        )
