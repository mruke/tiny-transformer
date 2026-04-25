from __future__ import annotations

import pytest
import torch

from tiny_transformer.model.masks import create_causal_attention_mask


# ===========================================================================
# create_causal_attention_mask tests
# ===========================================================================


# ---------------------------------------------------------------------------
# test_create_causal_attention_mask_returns_expected_shape
#
# This test checks that the mask has the correct square shape.
# ---------------------------------------------------------------------------
def test_create_causal_attention_mask_returns_expected_shape() -> None:
    mask = create_causal_attention_mask(sequence_length=4)

    assert mask.shape == (4, 4)


# ---------------------------------------------------------------------------
# test_create_causal_attention_mask_returns_boolean_tensor
#
# This test checks that the mask uses boolean values.
# ---------------------------------------------------------------------------
def test_create_causal_attention_mask_returns_boolean_tensor() -> None:
    mask = create_causal_attention_mask(sequence_length=4)

    assert mask.dtype == torch.bool


# ---------------------------------------------------------------------------
# test_create_causal_attention_mask_blocks_future_positions
#
# This test checks that positions above the diagonal are blocked.
# ---------------------------------------------------------------------------
def test_create_causal_attention_mask_blocks_future_positions() -> None:
    mask = create_causal_attention_mask(sequence_length=4)

    expected_mask = torch.tensor(
        [
            [False, True, True, True],
            [False, False, True, True],
            [False, False, False, True],
            [False, False, False, False],
        ],
        dtype=torch.bool,
    )

    assert torch.equal(mask, expected_mask)


# ---------------------------------------------------------------------------
# test_create_causal_attention_mask_allows_current_and_past_positions
#
# This test checks that the diagonal and lower triangle are not blocked.
# ---------------------------------------------------------------------------
def test_create_causal_attention_mask_allows_current_and_past_positions() -> None:
    mask = create_causal_attention_mask(sequence_length=5)

    # Lower triangle including the diagonal should be False.
    lower_triangle = torch.tril(mask)
    assert not lower_triangle.any()


# ---------------------------------------------------------------------------
# test_create_causal_attention_mask_rejects_invalid_sequence_length
#
# This test checks that sequence_length must be a positive integer.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad_sequence_length", [0, -1, "4"])
def test_create_causal_attention_mask_rejects_invalid_sequence_length(
    bad_sequence_length: int,
) -> None:
    with pytest.raises(ValueError, match="sequence_length must be a positive integer"):
        create_causal_attention_mask(bad_sequence_length)  # type: ignore[arg-type]
