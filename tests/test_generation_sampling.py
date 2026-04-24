from __future__ import annotations

import pytest
import torch

from tiny_transformer.inference.sampling import (
    greedy_sample_next_token,
    temperature_sample_next_token,
)


# ===========================================================================
# greedy_sample_next_token tests
# ===========================================================================


# ---------------------------------------------------------------------------
# test_greedy_sample_next_token_returns_expected_token_ids
#
# This test checks that greedy sampling picks the highest-logit token from
# each batch row.
# ---------------------------------------------------------------------------
def test_greedy_sample_next_token_returns_expected_token_ids() -> None:
    logits = torch.tensor(
        [
            [0.1, 0.8, 0.2],
            [1.2, 0.4, 0.3],
            [0.0, 0.2, 2.5],
        ]
    )

    next_token_ids = greedy_sample_next_token(logits)

    assert torch.equal(next_token_ids, torch.tensor([1, 0, 2]))


# ---------------------------------------------------------------------------
# test_greedy_sample_next_token_returns_1d_tensor
#
# This test checks that one token ID is returned per batch row.
# ---------------------------------------------------------------------------
def test_greedy_sample_next_token_returns_1d_tensor() -> None:
    logits = torch.randn(4, 6)

    next_token_ids = greedy_sample_next_token(logits)

    assert next_token_ids.dim() == 1
    assert next_token_ids.shape == (4,)


# ---------------------------------------------------------------------------
# test_greedy_sample_next_token_rejects_non_tensor_logits
#
# This test checks that logits must be a tensor.
# ---------------------------------------------------------------------------
def test_greedy_sample_next_token_rejects_non_tensor_logits() -> None:
    with pytest.raises(TypeError, match="logits must be a torch.Tensor"):
        greedy_sample_next_token([[0.1, 0.2, 0.3]])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# test_greedy_sample_next_token_rejects_non_2d_logits
#
# This test checks that logits must keep batch and vocab dimensions.
# ---------------------------------------------------------------------------
def test_greedy_sample_next_token_rejects_non_2d_logits() -> None:
    logits = torch.randn(3)

    with pytest.raises(
        ValueError,
        match="logits must have shape \\[batch_size, vocab_size\\]",
    ):
        greedy_sample_next_token(logits)


# ---------------------------------------------------------------------------
# test_greedy_sample_next_token_preserves_device
#
# This test checks that returned token IDs stay on the same device.
# ---------------------------------------------------------------------------
def test_greedy_sample_next_token_preserves_device() -> None:
    logits = torch.randn(2, 5)

    next_token_ids = greedy_sample_next_token(logits)

    assert next_token_ids.device == logits.device


# ===========================================================================
# temperature_sample_next_token tests
# ===========================================================================


# ---------------------------------------------------------------------------
# test_temperature_sample_next_token_returns_1d_tensor
#
# This test checks that one token ID is returned per batch row.
# ---------------------------------------------------------------------------
def test_temperature_sample_next_token_returns_1d_tensor() -> None:
    torch.manual_seed(0)
    logits = torch.randn(4, 6)

    next_token_ids = temperature_sample_next_token(
        logits=logits,
        temperature=1.0,
    )

    assert next_token_ids.dim() == 1
    assert next_token_ids.shape == (4,)


# ---------------------------------------------------------------------------
# test_temperature_sample_next_token_returns_valid_token_ids
#
# This test checks that sampled token IDs stay within the vocab range.
# ---------------------------------------------------------------------------
def test_temperature_sample_next_token_returns_valid_token_ids() -> None:
    torch.manual_seed(0)
    logits = torch.tensor(
        [
            [0.5, 1.0, 0.2],
            [1.4, 0.1, -0.2],
            [0.3, 0.7, 2.1],
        ]
    )

    next_token_ids = temperature_sample_next_token(
        logits=logits,
        temperature=1.0,
    )

    assert torch.all(next_token_ids >= 0)
    assert torch.all(next_token_ids < logits.shape[1])


# ---------------------------------------------------------------------------
# test_temperature_sample_next_token_rejects_non_tensor_logits
#
# This test checks that logits must be a tensor.
# ---------------------------------------------------------------------------
def test_temperature_sample_next_token_rejects_non_tensor_logits() -> None:
    with pytest.raises(TypeError, match="logits must be a torch.Tensor"):
        temperature_sample_next_token(
            logits=[[0.1, 0.2, 0.3]],  # type: ignore[arg-type]
            temperature=1.0,
        )


# ---------------------------------------------------------------------------
# test_temperature_sample_next_token_rejects_non_2d_logits
#
# This test checks that logits must keep batch and vocab dimensions.
# ---------------------------------------------------------------------------
def test_temperature_sample_next_token_rejects_non_2d_logits() -> None:
    logits = torch.randn(3)

    with pytest.raises(
        ValueError,
        match="logits must have shape \\[batch_size, vocab_size\\]",
    ):
        temperature_sample_next_token(
            logits=logits,
            temperature=1.0,
        )


# ---------------------------------------------------------------------------
# test_temperature_sample_next_token_rejects_non_numeric_temperature
#
# This test checks that temperature must be numeric.
# ---------------------------------------------------------------------------
def test_temperature_sample_next_token_rejects_non_numeric_temperature() -> None:
    logits = torch.randn(2, 5)

    with pytest.raises(TypeError, match="temperature must be a real number"):
        temperature_sample_next_token(
            logits=logits,
            temperature="1.0",  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# test_temperature_sample_next_token_rejects_non_positive_temperature
#
# This test checks that temperature must be greater than zero.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("temperature", [0.0, -1.0])
def test_temperature_sample_next_token_rejects_non_positive_temperature(
    temperature: float,
) -> None:
    logits = torch.randn(2, 5)

    with pytest.raises(ValueError, match="temperature must be greater than zero"):
        temperature_sample_next_token(
            logits=logits,
            temperature=temperature,
        )


# ---------------------------------------------------------------------------
# test_temperature_sample_next_token_preserves_device
#
# This test checks that returned token IDs stay on the same device.
# ---------------------------------------------------------------------------
def test_temperature_sample_next_token_preserves_device() -> None:
    torch.manual_seed(0)
    logits = torch.randn(2, 5)

    next_token_ids = temperature_sample_next_token(
        logits=logits,
        temperature=1.0,
    )

    assert next_token_ids.device == logits.device


# ---------------------------------------------------------------------------
# test_temperature_sample_next_token_rejects_non_integer_top_k
#
# This test checks that top_k must be an integer or None.
# ---------------------------------------------------------------------------
def test_temperature_sample_next_token_rejects_non_integer_top_k() -> None:
    logits = torch.randn(2, 5)

    with pytest.raises(TypeError, match="top_k must be an integer or None"):
        temperature_sample_next_token(
            logits=logits,
            temperature=1.0,
            top_k=2.5,  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# test_temperature_sample_next_token_rejects_non_positive_top_k
#
# This test checks that top_k must be greater than zero.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("top_k", [0, -1])
def test_temperature_sample_next_token_rejects_non_positive_top_k(
    top_k: int,
) -> None:
    logits = torch.randn(2, 5)

    with pytest.raises(ValueError, match="top_k must be greater than zero"):
        temperature_sample_next_token(
            logits=logits,
            temperature=1.0,
            top_k=top_k,
        )


# ---------------------------------------------------------------------------
# test_temperature_sample_next_token_rejects_top_k_above_vocab_size
#
# This test checks that top_k must not exceed vocab size.
# ---------------------------------------------------------------------------
def test_temperature_sample_next_token_rejects_top_k_above_vocab_size() -> None:
    logits = torch.randn(2, 5)

    with pytest.raises(ValueError, match="top_k must not exceed vocab_size"):
        temperature_sample_next_token(
            logits=logits,
            temperature=1.0,
            top_k=6,
        )


# ---------------------------------------------------------------------------
# test_temperature_sample_next_token_with_top_k_one_matches_greedy_choice
#
# This test checks that top_k=1 limits sampling to the single highest-logit
# token in each batch row.
# ---------------------------------------------------------------------------
def test_temperature_sample_next_token_with_top_k_one_matches_greedy_choice() -> None:
    torch.manual_seed(0)
    logits = torch.tensor(
        [
            [0.1, 0.8, 0.2],
            [1.2, 0.4, 0.3],
            [0.0, 0.2, 2.5],
        ]
    )

    sampled_token_ids = temperature_sample_next_token(
        logits=logits,
        temperature=1.0,
        top_k=1,
    )
    greedy_token_ids = greedy_sample_next_token(logits)

    assert torch.equal(sampled_token_ids, greedy_token_ids)


# ---------------------------------------------------------------------------
# test_temperature_sample_next_token_with_top_k_limits_possible_choices
#
# This test checks that top_k filtering prevents sampling from tokens outside
# the allowed top-k set.
# ---------------------------------------------------------------------------
def test_temperature_sample_next_token_with_top_k_limits_possible_choices() -> None:
    torch.manual_seed(0)
    logits = torch.tensor([[10.0, 9.0, -10.0, -10.0]])

    sampled_token_ids = temperature_sample_next_token(
        logits=logits,
        temperature=1.0,
        top_k=2,
    )

    assert sampled_token_ids.item() in {0, 1}
