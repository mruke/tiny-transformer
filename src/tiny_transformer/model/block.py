from __future__ import annotations

import torch
from torch import nn

from tiny_transformer.model.attention import MultiHeadSelfAttention
from tiny_transformer.model.feedforward import FeedForwardNetwork


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
# ---------------------------------------------------------------------------
def _validate_dropout_rate(dropout_rate: float) -> None:
    if not isinstance(dropout_rate, (int, float)) or not (0.0 <= dropout_rate < 1.0):
        raise ValueError("dropout_rate must be between 0.0 and 1.0.")


# ---------------------------------------------------------------------------
# _validate_hidden_states
#
# This function checks that hidden states are a 3D tensor.
# Expected shape: [batch_size, sequence_length, embedding_dim]
# ---------------------------------------------------------------------------
def _validate_hidden_states(hidden_states: torch.Tensor) -> None:
    if not isinstance(hidden_states, torch.Tensor):
        raise TypeError("hidden_states must be a torch.Tensor.")

    if hidden_states.dim() != 3:
        raise ValueError(
            "hidden_states must have shape "
            "[batch_size, sequence_length, embedding_dim]."
        )


# ---------------------------------------------------------------------------
# TransformerBlock
#
# TransformerBlock defines one decoder transformer block.
# It applies:
# - layer normalization
# - masked multi-head self-attention
# - residual connection
# - layer normalization
# - feedforward network
# - residual connection
# ---------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    # -----------------------------------------------------------------------
    # TransformerBlock.__init__
    #
    # This method creates the submodules needed for one transformer block.
    # -----------------------------------------------------------------------
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        feedforward_dim: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()

        _validate_positive_int(embedding_dim, "embedding_dim")
        _validate_positive_int(num_heads, "num_heads")
        _validate_positive_int(feedforward_dim, "feedforward_dim")
        _validate_dropout_rate(dropout_rate)

        self._attention_norm = nn.LayerNorm(embedding_dim)
        self._attention = MultiHeadSelfAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
        )
        self._attention_dropout = nn.Dropout(dropout_rate)

        self._feedforward_norm = nn.LayerNorm(embedding_dim)
        self._feedforward = FeedForwardNetwork(
            embedding_dim=embedding_dim,
            feedforward_dim=feedforward_dim,
            dropout_rate=dropout_rate,
        )
        self._feedforward_dropout = nn.Dropout(dropout_rate)

    # -----------------------------------------------------------------------
    # _apply_attention_stage
    #
    # This method applies the normalized attention stage and returns the
    # residual-updated hidden states.
    # -----------------------------------------------------------------------
    def _apply_attention_stage(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        normalized_hidden_states = self._attention_norm(hidden_states)
        attention_output = self._attention(normalized_hidden_states)
        attention_output = self._attention_dropout(attention_output)

        return hidden_states + attention_output

    # -----------------------------------------------------------------------
    # _apply_feedforward_stage
    #
    # This method applies the normalized feedforward stage and returns the
    # residual-updated hidden states.
    # -----------------------------------------------------------------------
    def _apply_feedforward_stage(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        normalized_hidden_states = self._feedforward_norm(hidden_states)
        feedforward_output = self._feedforward(normalized_hidden_states)
        feedforward_output = self._feedforward_dropout(feedforward_output)

        return hidden_states + feedforward_output

    # -----------------------------------------------------------------------
    # forward
    #
    # This method applies one full transformer block.
    # Input shape:  [batch_size, sequence_length, embedding_dim]
    # Output shape: [batch_size, sequence_length, embedding_dim]
    # -----------------------------------------------------------------------
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        _validate_hidden_states(hidden_states)

        hidden_states = self._apply_attention_stage(hidden_states)
        hidden_states = self._apply_feedforward_stage(hidden_states)

        return hidden_states
