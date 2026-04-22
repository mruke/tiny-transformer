from __future__ import annotations

from torch import nn


# ---------------------------------------------------------------------------
# _validate_positive_int
#
# This function checks that a value is a positive integer.
# ---------------------------------------------------------------------------
def _validate_positive_int(value: int, field_name: str) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer.")


# ---------------------------------------------------------------------------
# _validate_dropout_rate
#
# This function checks that dropout stays in the valid range.
# Dropout must be at least 0.0 and less than 1.0.
# ---------------------------------------------------------------------------
def _validate_dropout_rate(dropout_rate: float) -> None:
    if not isinstance(dropout_rate, (int, float)) or not (0.0 <= dropout_rate < 1.0):
        raise ValueError("dropout_rate must be between 0.0 and 1.0.")


# ---------------------------------------------------------------------------
# FeedForwardNetwork
#
# FeedForwardNetwork is the MLP block used inside a transformer block.
# It expands the embedding size, applies a non-linear activation, projects
# back down to the original embedding size, and applies dropout.
# ---------------------------------------------------------------------------
class FeedForwardNetwork(nn.Module):
    # -----------------------------------------------------------------------
    # FeedForwardNetwork.__init__
    #
    # This method builds the feedforward network layers.
    # -----------------------------------------------------------------------
    def __init__(
        self,
        embedding_dim: int,
        feedforward_dim: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()

        _validate_positive_int(embedding_dim, "embedding_dim")
        _validate_positive_int(feedforward_dim, "feedforward_dim")
        _validate_dropout_rate(dropout_rate)

        self._network = nn.Sequential(
            nn.Linear(embedding_dim, feedforward_dim),
            nn.GELU(),
            nn.Linear(feedforward_dim, embedding_dim),
            nn.Dropout(dropout_rate),
        )

    # -----------------------------------------------------------------------
    # forward
    #
    # This method applies the feedforward block to hidden states.
    # Input shape:  [batch_size, sequence_length, embedding_dim]
    # Output shape: [batch_size, sequence_length, embedding_dim]
    # -----------------------------------------------------------------------
    def forward(self, hidden_states):
        if hidden_states.dim() != 3:
            raise ValueError(
                "hidden_states must have shape "
                "[batch_size, sequence_length, embedding_dim]."
            )

        return self._network(hidden_states)
