from __future__ import annotations

import pytest

from tiny_transformer.data.dataset import TokenSequenceDataset
from tiny_transformer.data.splits import split_token_ids


# ===========================================================================
# split_token_ids tests
# ===========================================================================


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


# ===========================================================================
# TokenSequenceDataset tests
# ===========================================================================


# ---------------------------------------------------------------------------
# test_token_sequence_dataset_length_is_correct
#
# This test checks that dataset length matches the number of valid windows.
# ---------------------------------------------------------------------------
def test_token_sequence_dataset_length_is_correct() -> None:
    dataset = TokenSequenceDataset([0, 1, 2, 3, 4], context_window=2)

    assert len(dataset) == 3


# ---------------------------------------------------------------------------
# test_token_sequence_dataset_returns_shifted_input_target_pair
#
# This test checks that x and y are aligned for next-token prediction.
# ---------------------------------------------------------------------------
def test_token_sequence_dataset_returns_shifted_input_target_pair() -> None:
    dataset = TokenSequenceDataset([10, 11, 12, 13, 14], context_window=3)

    input_ids, target_ids = dataset[0]

    assert input_ids == [10, 11, 12]
    assert target_ids == [11, 12, 13]


# ---------------------------------------------------------------------------
# test_token_sequence_dataset_returns_correct_later_window
#
# This test checks that later dataset indices return later windows.
# ---------------------------------------------------------------------------
def test_token_sequence_dataset_returns_correct_later_window() -> None:
    dataset = TokenSequenceDataset([10, 11, 12, 13, 14], context_window=3)

    input_ids, target_ids = dataset[1]

    assert input_ids == [11, 12, 13]
    assert target_ids == [12, 13, 14]


# ---------------------------------------------------------------------------
# test_token_sequence_dataset_rejects_short_token_list
#
# This test checks that the token list must be longer than the context window.
# ---------------------------------------------------------------------------
def test_token_sequence_dataset_rejects_short_token_list() -> None:
    with pytest.raises(ValueError, match="longer than the context window"):
        TokenSequenceDataset([1, 2, 3], context_window=3)


# ---------------------------------------------------------------------------
# test_token_sequence_dataset_rejects_invalid_context_window
#
# This test checks that context window must be positive.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad_context_window", [0, -1])
def test_token_sequence_dataset_rejects_invalid_context_window(
    bad_context_window: int,
) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        TokenSequenceDataset([1, 2, 3, 4], context_window=bad_context_window)


# ---------------------------------------------------------------------------
# test_token_sequence_dataset_rejects_out_of_range_index
#
# This test checks that invalid indices fail clearly.
# ---------------------------------------------------------------------------
def test_token_sequence_dataset_rejects_out_of_range_index() -> None:
    dataset = TokenSequenceDataset([1, 2, 3, 4], context_window=2)

    with pytest.raises(IndexError, match="out of range"):
        _ = dataset[2]


# ---------------------------------------------------------------------------
# test_token_sequence_dataset_rejects_non_integer_index
#
# This test checks that dataset indices must be integers.
# ---------------------------------------------------------------------------
def test_token_sequence_dataset_rejects_non_integer_index() -> None:
    dataset = TokenSequenceDataset([1, 2, 3, 4], context_window=2)

    with pytest.raises(TypeError, match="must be an integer"):
        _ = dataset["0"]  # type: ignore[index]
