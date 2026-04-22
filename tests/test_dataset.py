from __future__ import annotations

import pytest

from tiny_transformer.data.splits import (
    _validate_token_ids,
    _validate_train_split_ratio,
    split_token_ids,
)


# ---------------------------------------------------------------------------
# test_split_token_ids_returns_expected_train_and_validation_splits
#
# This test checks that token IDs are split in the expected place.
# ---------------------------------------------------------------------------
def test_split_token_ids_returns_expected_train_and_validation_splits() -> None:
    token_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    train_token_ids, validation_token_ids = split_token_ids(token_ids, 0.8)

    assert train_token_ids == [0, 1, 2, 3, 4, 5, 6, 7]
    assert validation_token_ids == [8, 9]


# ---------------------------------------------------------------------------
# test_split_token_ids_preserves_all_tokens
#
# This test checks that no token IDs are lost during the split.
# ---------------------------------------------------------------------------
def test_split_token_ids_preserves_all_tokens() -> None:
    token_ids = [10, 11, 12, 13, 14, 15]

    train_token_ids, validation_token_ids = split_token_ids(token_ids, 0.5)

    assert train_token_ids + validation_token_ids == token_ids


# ---------------------------------------------------------------------------
# test_split_token_ids_rejects_empty_list
#
# This test checks that an empty token list fails clearly.
# ---------------------------------------------------------------------------
def test_split_token_ids_rejects_empty_list() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        split_token_ids([], 0.8)


# ---------------------------------------------------------------------------
# test_split_token_ids_rejects_invalid_ratio
#
# This test checks that the split ratio must stay between 0 and 1.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad_ratio", [0.0, 1.0, -0.5, 1.5])
def test_split_token_ids_rejects_invalid_ratio(bad_ratio: float) -> None:
    token_ids = [0, 1, 2, 3]

    with pytest.raises(ValueError, match="greater than 0 and less than 1"):
        split_token_ids(token_ids, bad_ratio)


# ---------------------------------------------------------------------------
# test_split_token_ids_rejects_non_integer_token_ids
#
# This test checks that every token ID must be an integer.
# ---------------------------------------------------------------------------
def test_split_token_ids_rejects_non_integer_token_ids() -> None:
    token_ids = [0, 1, "2", 3]

    with pytest.raises(TypeError, match="must be an integer"):
        split_token_ids(token_ids, 0.8)


# ---------------------------------------------------------------------------
# test_split_token_ids_rejects_split_that_empties_one_side
#
# This test checks that both train and validation splits must keep data.
# ---------------------------------------------------------------------------
def test_split_token_ids_rejects_split_that_empties_one_side() -> None:
    token_ids = [0]

    with pytest.raises(ValueError, match="leave at least one token"):
        split_token_ids(token_ids, 0.8)


# ---------------------------------------------------------------------------
# test_validate_token_ids_accepts_valid_list
#
# This test checks that valid token IDs pass validation.
# ---------------------------------------------------------------------------
def test_validate_token_ids_accepts_valid_list() -> None:
    _validate_token_ids([0, 1, 2, 3])


# ---------------------------------------------------------------------------
# test_validate_token_ids_rejects_non_list_input
#
# This test checks that token IDs must be provided as a list.
# ---------------------------------------------------------------------------
def test_validate_token_ids_rejects_non_list_input() -> None:
    with pytest.raises(TypeError, match="provided as a list"):
        _validate_token_ids("not-a-list")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# test_validate_train_split_ratio_accepts_valid_ratio
#
# This test checks that a valid split ratio passes validation.
# ---------------------------------------------------------------------------
def test_validate_train_split_ratio_accepts_valid_ratio() -> None:
    _validate_train_split_ratio(0.8)


# ---------------------------------------------------------------------------
# test_validate_train_split_ratio_rejects_bad_values
#
# This test checks that invalid split ratio values fail clearly.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad_ratio", [0.0, 1.0, -1.0, 2.0, "0.8"])
def test_validate_train_split_ratio_rejects_bad_values(
    bad_ratio: float,
) -> None:
    with pytest.raises(ValueError, match="greater than 0 and less than 1"):
        _validate_train_split_ratio(bad_ratio)  # type: ignore[arg-type]
