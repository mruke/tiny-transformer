from __future__ import annotations


# ---------------------------------------------------------------------------
# _validate_token_ids
#
# This function checks that token IDs are provided as a non-empty list of
# integers. This keeps low-level validation separate from dataset behavior.
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
# _validate_context_window
#
# This function checks that the context window is a positive integer.
# ---------------------------------------------------------------------------
def _validate_context_window(context_window: int) -> None:
    if not isinstance(context_window, int) or context_window <= 0:
        raise ValueError("Context window must be a positive integer.")


# ---------------------------------------------------------------------------
# _validate_dataset_size
#
# This function checks that the token list is long enough to create at least
# one input-target sample.
# ---------------------------------------------------------------------------
def _validate_dataset_size(
    token_ids: list[int],
    context_window: int,
) -> None:
    if len(token_ids) <= context_window:
        raise ValueError("Token ID list must be longer than the context window.")


# ---------------------------------------------------------------------------
# TokenSequenceDataset
#
# TokenSequenceDataset turns encoded token IDs into fixed-length input and
# target pairs for next-token prediction.
# ---------------------------------------------------------------------------
class TokenSequenceDataset:
    # -----------------------------------------------------------------------
    # TokenSequenceDataset.__init__
    #
    # This method stores token IDs and the context window size after
    # validation has passed.
    # -----------------------------------------------------------------------
    def __init__(self, token_ids: list[int], context_window: int) -> None:
        _validate_token_ids(token_ids)
        _validate_context_window(context_window)
        _validate_dataset_size(token_ids, context_window)

        self._token_ids = token_ids
        self._context_window = context_window

    # -----------------------------------------------------------------------
    # context_window
    #
    # This property returns the configured context window size.
    # -----------------------------------------------------------------------
    @property
    def context_window(self) -> int:
        return self._context_window

    # -----------------------------------------------------------------------
    # token_ids
    #
    # This property returns a copy of the stored token IDs.
    # A copy is returned so outside code cannot change dataset state.
    # -----------------------------------------------------------------------
    @property
    def token_ids(self) -> list[int]:
        return list(self._token_ids)

    # -----------------------------------------------------------------------
    # __len__
    #
    # This method returns the number of valid sample starting positions.
    # -----------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._token_ids) - self._context_window

    # -----------------------------------------------------------------------
    # __getitem__
    #
    # This method returns one input-target pair.
    # The target is the input window shifted by one token.
    # -----------------------------------------------------------------------
    def __getitem__(self, index: int) -> tuple[list[int], list[int]]:
        if not isinstance(index, int):
            raise TypeError("Dataset index must be an integer.")

        if index < 0 or index >= len(self):
            raise IndexError("Dataset index is out of range.")

        start_index = index
        end_index = index + self._context_window

        input_ids = self._token_ids[start_index:end_index]
        target_ids = self._token_ids[start_index + 1 : end_index + 1]

        return input_ids, target_ids
