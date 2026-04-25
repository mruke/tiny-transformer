from __future__ import annotations

import pytest
import torch

from tiny_transformer.model.block import TransformerBlock


# ===========================================================================
# TransformerBlock tests
# ===========================================================================


# ---------------------------------------------------------------------------
# test_transformer_block_returns_expected_shape
#
# This test checks that the transformer block preserves the outer tensor shape.
# ---------------------------------------------------------------------------
def test_transformer_block_returns_expected_shape() -> None:
    block = TransformerBlock(
        embedding_dim=16,
        num_heads=4,
        feedforward_dim=64,
        dropout_rate=0.1,
    )
    hidden_states = torch.randn(2, 5, 16)

    output = block(hidden_states)

    assert output.shape == (2, 5, 16)


# ---------------------------------------------------------------------------
# test_transformer_block_rejects_invalid_embedding_dim
#
# This test checks that embedding_dim must be a positive integer.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad_embedding_dim", [0, -1, "16"])
def test_transformer_block_rejects_invalid_embedding_dim(
    bad_embedding_dim: int,
) -> None:
    with pytest.raises(ValueError, match="embedding_dim must be a positive integer"):
        TransformerBlock(
            embedding_dim=bad_embedding_dim,  # type: ignore[arg-type]
            num_heads=4,
            feedforward_dim=64,
            dropout_rate=0.1,
        )


# ---------------------------------------------------------------------------
# test_transformer_block_rejects_invalid_num_heads
#
# This test checks that num_heads must be a positive integer.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad_num_heads", [0, -1, "4"])
def test_transformer_block_rejects_invalid_num_heads(
    bad_num_heads: int,
) -> None:
    with pytest.raises(ValueError, match="num_heads must be a positive integer"):
        TransformerBlock(
            embedding_dim=16,
            num_heads=bad_num_heads,  # type: ignore[arg-type]
            feedforward_dim=64,
            dropout_rate=0.1,
        )


# ---------------------------------------------------------------------------
# test_transformer_block_rejects_invalid_feedforward_dim
#
# This test checks that feedforward_dim must be a positive integer.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad_feedforward_dim", [0, -1, "64"])
def test_transformer_block_rejects_invalid_feedforward_dim(
    bad_feedforward_dim: int,
) -> None:
    with pytest.raises(ValueError, match="feedforward_dim must be a positive integer"):
        TransformerBlock(
            embedding_dim=16,
            num_heads=4,
            feedforward_dim=bad_feedforward_dim,  # type: ignore[arg-type]
            dropout_rate=0.1,
        )


# ---------------------------------------------------------------------------
# test_transformer_block_rejects_invalid_dropout_rate
#
# This test checks that dropout_rate must stay in the valid range.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad_dropout_rate", [-0.1, 1.0, 1.5, "0.1"])
def test_transformer_block_rejects_invalid_dropout_rate(
    bad_dropout_rate: float,
) -> None:
    with pytest.raises(ValueError, match="dropout_rate must be between 0.0 and 1.0"):
        TransformerBlock(
            embedding_dim=16,
            num_heads=4,
            feedforward_dim=64,
            dropout_rate=bad_dropout_rate,  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# test_transformer_block_rejects_non_tensor_input
#
# This test checks that hidden_states must be a torch tensor.
# ---------------------------------------------------------------------------
def test_transformer_block_rejects_non_tensor_input() -> None:
    block = TransformerBlock(
        embedding_dim=16,
        num_heads=4,
        feedforward_dim=64,
        dropout_rate=0.1,
    )

    with pytest.raises(TypeError, match="hidden_states must be a torch.Tensor"):
        block([[1, 2, 3]])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# test_transformer_block_rejects_non_3d_input
#
# This test checks that hidden_states must be a 3D tensor.
# ---------------------------------------------------------------------------
def test_transformer_block_rejects_non_3d_input() -> None:
    block = TransformerBlock(
        embedding_dim=16,
        num_heads=4,
        feedforward_dim=64,
        dropout_rate=0.1,
    )
    hidden_states = torch.randn(5, 16)

    with pytest.raises(ValueError, match="hidden_states must have shape"):
        block(hidden_states)


# ---------------------------------------------------------------------------
# test_transformer_block_returns_finite_values
#
# This test checks that the output does not contain NaN or infinite values for
# a normal input tensor.
# ---------------------------------------------------------------------------
def test_transformer_block_returns_finite_values() -> None:
    block = TransformerBlock(
        embedding_dim=16,
        num_heads=4,
        feedforward_dim=64,
        dropout_rate=0.0,
    )
    hidden_states = torch.randn(2, 5, 16)

    output = block(hidden_states)

    assert torch.isfinite(output).all()


# ---------------------------------------------------------------------------
# test_transformer_block_preserves_device
#
# This test checks that the output stays on the same device as the input.
# ---------------------------------------------------------------------------
def test_transformer_block_preserves_device() -> None:
    block = TransformerBlock(
        embedding_dim=16,
        num_heads=4,
        feedforward_dim=64,
        dropout_rate=0.0,
    )
    hidden_states = torch.randn(2, 5, 16)

    output = block(hidden_states)

    assert output.device == hidden_states.device


# ---------------------------------------------------------------------------
# test_transformer_block_supports_single_token_sequence
#
# This test checks that a sequence length of one still works.
# ---------------------------------------------------------------------------
def test_transformer_block_supports_single_token_sequence() -> None:
    block = TransformerBlock(
        embedding_dim=16,
        num_heads=4,
        feedforward_dim=64,
        dropout_rate=0.0,
    )
    hidden_states = torch.randn(2, 1, 16)

    output = block(hidden_states)

    assert output.shape == (2, 1, 16)
    assert torch.isfinite(output).all()
