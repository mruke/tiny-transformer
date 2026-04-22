from __future__ import annotations


# ---------------------------------------------------------------------------
# _validate_token_ids
#
# This function checks that token IDs are provided as a non-empty list of
# integers. This keeps low-level validation separate from split logic.
# ---------------------------------------------------------------------------
def _validate_token_ids(token_ids: list[int]) -> None:
    if not isinstance(token_ids, list):
        raise TypeError("Token IDs must be provided as a list.")

    if not token_ids:
        raise ValueError("Token ID list cannot be empty.")

    for token_id in token_ids:
        if not isinstance(token_id, int):
            raise TypeError("Each token ID must be an integer.")


# ---------------------------------------------------------------------------
# _validate_train_split_ratio
#
# This function checks that the train split ratio is between 0 and 1.
# ---------------------------------------------------------------------------
def _validate_train_split_ratio(train_split_ratio: float) -> None:
    if not isinstance(train_split_ratio, (int, float)) or not (
        0.0 < train_split_ratio < 1.0
    ):
        raise ValueError("Train split ratio must be greater than 0 and less than 1.")


# ---------------------------------------------------------------------------
# split_token_ids
#
# This function splits encoded token IDs into training and validation parts.
# The split ratio decides how much data goes into the training portion.
# ---------------------------------------------------------------------------
def split_token_ids(
    token_ids: list[int],
    train_split_ratio: float,
) -> tuple[list[int], list[int]]:
    _validate_token_ids(token_ids)
    _validate_train_split_ratio(train_split_ratio)

    split_index = int(len(token_ids) * train_split_ratio)

    if split_index <= 0 or split_index >= len(token_ids):
        raise ValueError(
            "Train split ratio must leave at least one token in both splits."
        )

    train_token_ids = token_ids[:split_index]
    validation_token_ids = token_ids[split_index:]

    return train_token_ids, validation_token_ids
