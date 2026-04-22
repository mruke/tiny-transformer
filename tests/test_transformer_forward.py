from __future__ import annotations

import pytest
import torch

from tiny_transformer.model.embeddings import (
    PositionalEmbedding,
    TokenEmbedding,
    TransformerEmbedding,
)
from tiny_transformer.model.feedforward import FeedForwardNetwork


# ===========================================================================
# TokenEmbedding tests
# ===========================================================================


# ---------------------------------------------------------------------------
# test_token_embedding_returns_expected_shape
#
# This test checks that token embeddings return the correct 3D output shape.
# ---------------------------------------------------------------------------
def test_token_embedding_returns_expected_shape() -> None:
    embedding = TokenEmbedding(vocab_size=30, embedding_dim=16)
    token_ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)

    output = embedding(token_ids)

    assert output.shape == (2, 3, 16)


# ---------------------------------------------------------------------------
# test_token_embedding_rejects_invalid_vocab_size
#
# This test checks that vocab_size must be a positive integer.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad_vocab_size", [0, -1, "10"])
def test_token_embedding_rejects_invalid_vocab_size(
    bad_vocab_size: int,
) -> None:
    with pytest.raises(ValueError, match="vocab_size must be a positive integer"):
        TokenEmbedding(vocab_size=bad_vocab_size, embedding_dim=16)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# test_token_embedding_rejects_invalid_embedding_dim
#
# This test checks that embedding_dim must be a positive integer.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad_embedding_dim", [0, -1, "16"])
def test_token_embedding_rejects_invalid_embedding_dim(
    bad_embedding_dim: int,
) -> None:
    with pytest.raises(ValueError, match="embedding_dim must be a positive integer"):
        TokenEmbedding(vocab_size=30, embedding_dim=bad_embedding_dim)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# test_token_embedding_rejects_non_2d_input
#
# This test checks that token IDs must have shape
# [batch_size, sequence_length].
# ---------------------------------------------------------------------------
def test_token_embedding_rejects_non_2d_input() -> None:
    embedding = TokenEmbedding(vocab_size=30, embedding_dim=16)
    token_ids = torch.tensor([1, 2, 3], dtype=torch.long)

    with pytest.raises(ValueError, match="token_ids must have shape"):
        embedding(token_ids)


# ===========================================================================
# PositionalEmbedding tests
# ===========================================================================


# ---------------------------------------------------------------------------
# test_positional_embedding_returns_expected_shape
#
# This test checks that positional embeddings return the correct 3D output
# shape for a batch.
# ---------------------------------------------------------------------------
def test_positional_embedding_returns_expected_shape() -> None:
    embedding = PositionalEmbedding(max_sequence_length=8, embedding_dim=16)

    output = embedding(
        batch_size=2,
        sequence_length=4,
        device=torch.device("cpu"),
    )

    assert output.shape == (2, 4, 16)


# ---------------------------------------------------------------------------
# test_positional_embedding_rejects_invalid_max_sequence_length
#
# This test checks that max_sequence_length must be a positive integer.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad_max_sequence_length", [0, -1, "8"])
def test_positional_embedding_rejects_invalid_max_sequence_length(
    bad_max_sequence_length: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="max_sequence_length must be a positive integer",
    ):
        PositionalEmbedding(
            max_sequence_length=bad_max_sequence_length,  # type: ignore[arg-type]
            embedding_dim=16,
        )


# ---------------------------------------------------------------------------
# test_positional_embedding_rejects_invalid_embedding_dim
#
# This test checks that embedding_dim must be a positive integer.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad_embedding_dim", [0, -1, "16"])
def test_positional_embedding_rejects_invalid_embedding_dim(
    bad_embedding_dim: int,
) -> None:
    with pytest.raises(ValueError, match="embedding_dim must be a positive integer"):
        PositionalEmbedding(
            max_sequence_length=8,
            embedding_dim=bad_embedding_dim,  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# test_positional_embedding_rejects_sequence_length_above_max
#
# This test checks that sequence_length cannot be larger than the configured
# maximum sequence length.
# ---------------------------------------------------------------------------
def test_positional_embedding_rejects_sequence_length_above_max() -> None:
    embedding = PositionalEmbedding(max_sequence_length=4, embedding_dim=16)

    with pytest.raises(ValueError, match="cannot exceed max_sequence_length"):
        embedding(
            batch_size=2,
            sequence_length=5,
            device=torch.device("cpu"),
        )


# ---------------------------------------------------------------------------
# test_positional_embedding_rejects_invalid_batch_size
#
# This test checks that batch_size must be a positive integer.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad_batch_size", [0, -1, "2"])
def test_positional_embedding_rejects_invalid_batch_size(
    bad_batch_size: int,
) -> None:
    embedding = PositionalEmbedding(max_sequence_length=8, embedding_dim=16)

    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        embedding(
            batch_size=bad_batch_size,  # type: ignore[arg-type]
            sequence_length=4,
            device=torch.device("cpu"),
        )


# ---------------------------------------------------------------------------
# test_positional_embedding_rejects_invalid_sequence_length
#
# This test checks that sequence_length must be a positive integer.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad_sequence_length", [0, -1, "4"])
def test_positional_embedding_rejects_invalid_sequence_length(
    bad_sequence_length: int,
) -> None:
    embedding = PositionalEmbedding(max_sequence_length=8, embedding_dim=16)

    with pytest.raises(ValueError, match="sequence_length must be a positive integer"):
        embedding(
            batch_size=2,
            sequence_length=bad_sequence_length,  # type: ignore[arg-type]
            device=torch.device("cpu"),
        )


# ===========================================================================
# TransformerEmbedding tests
# ===========================================================================


# ---------------------------------------------------------------------------
# test_transformer_embedding_returns_expected_shape
#
# This test checks that combined token and positional embeddings return the
# expected output shape.
# ---------------------------------------------------------------------------
def test_transformer_embedding_returns_expected_shape() -> None:
    embedding = TransformerEmbedding(
        vocab_size=50,
        max_sequence_length=8,
        embedding_dim=16,
    )
    token_ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)

    output = embedding(token_ids)

    assert output.shape == (2, 3, 16)


# ---------------------------------------------------------------------------
# test_transformer_embedding_rejects_non_2d_input
#
# This test checks that combined embeddings require token IDs with shape
# [batch_size, sequence_length].
# ---------------------------------------------------------------------------
def test_transformer_embedding_rejects_non_2d_input() -> None:
    embedding = TransformerEmbedding(
        vocab_size=50,
        max_sequence_length=8,
        embedding_dim=16,
    )
    token_ids = torch.tensor([1, 2, 3], dtype=torch.long)

    with pytest.raises(ValueError, match="token_ids must have shape"):
        embedding(token_ids)


# ===========================================================================
# FeedForwardNetwork tests
# ===========================================================================


# ---------------------------------------------------------------------------
# test_feedforward_network_returns_expected_shape
#
# This test checks that the feedforward network preserves the outer tensor
# shape and returns the original embedding size.
# ---------------------------------------------------------------------------
def test_feedforward_network_returns_expected_shape() -> None:
    network = FeedForwardNetwork(
        embedding_dim=16,
        feedforward_dim=64,
        dropout_rate=0.1,
    )
    hidden_states = torch.randn(2, 4, 16)

    output = network(hidden_states)

    assert output.shape == (2, 4, 16)


# ---------------------------------------------------------------------------
# test_feedforward_network_rejects_invalid_embedding_dim
#
# This test checks that embedding_dim must be a positive integer.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad_embedding_dim", [0, -1, "16"])
def test_feedforward_network_rejects_invalid_embedding_dim(
    bad_embedding_dim: int,
) -> None:
    with pytest.raises(ValueError, match="embedding_dim must be a positive integer"):
        FeedForwardNetwork(
            embedding_dim=bad_embedding_dim,  # type: ignore[arg-type]
            feedforward_dim=64,
            dropout_rate=0.1,
        )


# ---------------------------------------------------------------------------
# test_feedforward_network_rejects_invalid_feedforward_dim
#
# This test checks that feedforward_dim must be a positive integer.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad_feedforward_dim", [0, -1, "64"])
def test_feedforward_network_rejects_invalid_feedforward_dim(
    bad_feedforward_dim: int,
) -> None:
    with pytest.raises(ValueError, match="feedforward_dim must be a positive integer"):
        FeedForwardNetwork(
            embedding_dim=16,
            feedforward_dim=bad_feedforward_dim,  # type: ignore[arg-type]
            dropout_rate=0.1,
        )


# ---------------------------------------------------------------------------
# test_feedforward_network_rejects_invalid_dropout_rate
#
# This test checks that dropout_rate must stay in the valid range.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad_dropout_rate", [-0.1, 1.0, 1.5, "0.1"])
def test_feedforward_network_rejects_invalid_dropout_rate(
    bad_dropout_rate: float,
) -> None:
    with pytest.raises(ValueError, match="dropout_rate must be between 0.0 and 1.0"):
        FeedForwardNetwork(
            embedding_dim=16,
            feedforward_dim=64,
            dropout_rate=bad_dropout_rate,  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# test_feedforward_network_rejects_non_3d_input
#
# This test checks that hidden_states must have shape
# [batch_size, sequence_length, embedding_dim].
# ---------------------------------------------------------------------------
def test_feedforward_network_rejects_non_3d_input() -> None:
    network = FeedForwardNetwork(
        embedding_dim=16,
        feedforward_dim=64,
        dropout_rate=0.1,
    )
    hidden_states = torch.randn(4, 16)

    with pytest.raises(ValueError, match="hidden_states must have shape"):
        network(hidden_states)
