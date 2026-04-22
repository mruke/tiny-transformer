from __future__ import annotations

import torch
from torch import nn

from tiny_transformer.model.block import TransformerBlock
from tiny_transformer.model.embeddings import TransformerEmbedding


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
# _validate_token_ids
#
# This function checks that token IDs are provided as a 2D tensor.
# Expected shape: [batch_size, sequence_length]
# ---------------------------------------------------------------------------
def _validate_token_ids(token_ids: torch.Tensor) -> None:
    if not isinstance(token_ids, torch.Tensor):
        raise TypeError("token_ids must be a torch.Tensor.")

    if token_ids.dim() != 2:
        raise ValueError("token_ids must have shape [batch_size, sequence_length].")


# ---------------------------------------------------------------------------
# DecoderOnlyTransformer
#
# DecoderOnlyTransformer defines the top-level language model for this project.
# It applies:
# - token and positional embeddings
# - a stack of transformer blocks
# - final normalization
# - projection to vocabulary logits
# ---------------------------------------------------------------------------
class DecoderOnlyTransformer(nn.Module):
    # -----------------------------------------------------------------------
    # DecoderOnlyTransformer.__init__
    #
    # This method creates the embedding stage, transformer block stack,
    # final normalization, and output projection.
    # -----------------------------------------------------------------------
    def __init__(
        self,
        vocab_size: int,
        max_sequence_length: int,
        embedding_dim: int,
        num_heads: int,
        num_layers: int,
        feedforward_dim: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()

        _validate_positive_int(vocab_size, "vocab_size")
        _validate_positive_int(max_sequence_length, "max_sequence_length")
        _validate_positive_int(embedding_dim, "embedding_dim")
        _validate_positive_int(num_heads, "num_heads")
        _validate_positive_int(num_layers, "num_layers")
        _validate_positive_int(feedforward_dim, "feedforward_dim")
        _validate_dropout_rate(dropout_rate)

        self._vocab_size = vocab_size
        self._max_sequence_length = max_sequence_length
        self._embedding_dim = embedding_dim

        self._embedding = TransformerEmbedding(
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            embedding_dim=embedding_dim,
        )

        self._blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    feedforward_dim=feedforward_dim,
                    dropout_rate=dropout_rate,
                )
                for _ in range(num_layers)
            ]
        )

        self._final_norm = nn.LayerNorm(embedding_dim)
        self._output_projection = nn.Linear(embedding_dim, vocab_size)

    # -----------------------------------------------------------------------
    # vocab_size
    #
    # This property returns the configured vocabulary size.
    # -----------------------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    # -----------------------------------------------------------------------
    # max_sequence_length
    #
    # This property returns the configured maximum supported sequence length.
    # -----------------------------------------------------------------------
    @property
    def max_sequence_length(self) -> int:
        return self._max_sequence_length

    # -----------------------------------------------------------------------
    # _validate_sequence_length
    #
    # This method checks that the input sequence length does not exceed the
    # configured maximum supported sequence length.
    # -----------------------------------------------------------------------
    def _validate_sequence_length(self, token_ids: torch.Tensor) -> None:
        _, sequence_length = token_ids.shape

        if sequence_length > self._max_sequence_length:
            raise ValueError("Input sequence length cannot exceed max_sequence_length.")

    # -----------------------------------------------------------------------
    # _apply_transformer_blocks
    #
    # This method applies the full stack of transformer blocks.
    # -----------------------------------------------------------------------
    def _apply_transformer_blocks(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        for block in self._blocks:
            hidden_states = block(hidden_states)

        return hidden_states

    # -----------------------------------------------------------------------
    # forward
    #
    # This method applies the full decoder-only transformer forward pass.
    # Input shape:  [batch_size, sequence_length]
    # Output shape: [batch_size, sequence_length, vocab_size]
    # -----------------------------------------------------------------------
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        _validate_token_ids(token_ids)
        self._validate_sequence_length(token_ids)

        hidden_states = self._embedding(token_ids)
        hidden_states = self._apply_transformer_blocks(hidden_states)
        hidden_states = self._final_norm(hidden_states)

        logits = self._output_projection(hidden_states)

        return logits
