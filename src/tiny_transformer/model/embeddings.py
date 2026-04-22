from __future__ import annotations

import torch
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
# _validate_token_id_tensor
#
# This function checks that token IDs are provided as a 2D tensor.
# Expected shape: [batch_size, sequence_length]
# ---------------------------------------------------------------------------
def _validate_token_id_tensor(token_ids: torch.Tensor) -> None:
    if not isinstance(token_ids, torch.Tensor):
        raise TypeError("token_ids must be a torch.Tensor.")

    if token_ids.dim() != 2:
        raise ValueError("token_ids must have shape [batch_size, sequence_length].")


# ---------------------------------------------------------------------------
# TokenEmbedding
#
# TokenEmbedding turns token IDs into learned vectors.
# Each token ID maps to one row in the embedding table.
# ---------------------------------------------------------------------------
class TokenEmbedding(nn.Module):
    # -----------------------------------------------------------------------
    # TokenEmbedding.__init__
    #
    # This method creates the token embedding table.
    # -----------------------------------------------------------------------
    def __init__(self, vocab_size: int, embedding_dim: int) -> None:
        super().__init__()

        _validate_positive_int(vocab_size, "vocab_size")
        _validate_positive_int(embedding_dim, "embedding_dim")

        self._embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
        )

    # -----------------------------------------------------------------------
    # forward
    #
    # This method maps token IDs to embedding vectors.
    # Input shape:  [batch_size, sequence_length]
    # Output shape: [batch_size, sequence_length, embedding_dim]
    # -----------------------------------------------------------------------
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        _validate_token_id_tensor(token_ids)
        return self._embedding(token_ids)


# ---------------------------------------------------------------------------
# PositionalEmbedding
#
# PositionalEmbedding creates learned position vectors.
# This gives the model a way to know where each token sits in the sequence.
# ---------------------------------------------------------------------------
class PositionalEmbedding(nn.Module):
    # -----------------------------------------------------------------------
    # PositionalEmbedding.__init__
    #
    # This method creates the position embedding table.
    # -----------------------------------------------------------------------
    def __init__(self, max_sequence_length: int, embedding_dim: int) -> None:
        super().__init__()

        _validate_positive_int(max_sequence_length, "max_sequence_length")
        _validate_positive_int(embedding_dim, "embedding_dim")

        self._max_sequence_length = max_sequence_length
        self._embedding = nn.Embedding(
            num_embeddings=max_sequence_length,
            embedding_dim=embedding_dim,
        )

    # -----------------------------------------------------------------------
    # forward
    #
    # This method creates position embeddings for a batch.
    # Output shape: [batch_size, sequence_length, embedding_dim]
    # -----------------------------------------------------------------------
    def forward(
        self,
        batch_size: int,
        sequence_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        _validate_positive_int(batch_size, "batch_size")
        _validate_positive_int(sequence_length, "sequence_length")

        if sequence_length > self._max_sequence_length:
            raise ValueError("sequence_length cannot exceed max_sequence_length.")

        # Position IDs are built as [0, 1, 2, ..., sequence_length - 1].
        position_ids = torch.arange(
            sequence_length,
            device=device,
            dtype=torch.long,
        )

        # The same position pattern is used for every item in the batch.
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        return self._embedding(position_ids)


# ---------------------------------------------------------------------------
# TransformerEmbedding
#
# TransformerEmbedding combines token embeddings and position embeddings.
# This is the full embedding stage used before attention blocks.
# ---------------------------------------------------------------------------
class TransformerEmbedding(nn.Module):
    # -----------------------------------------------------------------------
    # TransformerEmbedding.__init__
    #
    # This method creates the token and position embedding modules.
    # -----------------------------------------------------------------------
    def __init__(
        self,
        vocab_size: int,
        max_sequence_length: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()

        self._token_embedding = TokenEmbedding(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
        )
        self._positional_embedding = PositionalEmbedding(
            max_sequence_length=max_sequence_length,
            embedding_dim=embedding_dim,
        )

    # -----------------------------------------------------------------------
    # forward
    #
    # This method adds token embeddings and position embeddings together.
    # Input shape:  [batch_size, sequence_length]
    # Output shape: [batch_size, sequence_length, embedding_dim]
    # -----------------------------------------------------------------------
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        _validate_token_id_tensor(token_ids)

        batch_size, sequence_length = token_ids.shape

        token_vectors = self._token_embedding(token_ids)
        position_vectors = self._positional_embedding(
            batch_size=batch_size,
            sequence_length=sequence_length,
            device=token_ids.device,
        )

        return token_vectors + position_vectors
