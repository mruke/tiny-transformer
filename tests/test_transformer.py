from __future__ import annotations

import pytest
import torch

from tiny_transformer.model.transformer import DecoderOnlyTransformer


# ===========================================================================
# DecoderOnlyTransformer tests
# ===========================================================================


# ---------------------------------------------------------------------------
# test_decoder_only_transformer_returns_expected_logits_shape
#
# This test checks that the top-level model returns logits with shape
# [batch_size, sequence_length, vocab_size].
# ---------------------------------------------------------------------------
def test_decoder_only_transformer_returns_expected_logits_shape() -> None:
    model = DecoderOnlyTransformer(
        vocab_size=30,
        max_sequence_length=8,
        embedding_dim=16,
        num_heads=4,
        num_layers=2,
        feedforward_dim=64,
        dropout_rate=0.1,
    )
    token_ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)

    logits = model(token_ids)

    assert logits.shape == (2, 3, 30)


# ---------------------------------------------------------------------------
# test_decoder_only_transformer_rejects_invalid_vocab_size
#
# This test checks that vocab_size must be a positive integer.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad_vocab_size", [0, -1, "30"])
def test_decoder_only_transformer_rejects_invalid_vocab_size(
    bad_vocab_size: int,
) -> None:
    with pytest.raises(ValueError, match="vocab_size must be a positive integer"):
        DecoderOnlyTransformer(
            vocab_size=bad_vocab_size,  # type: ignore[arg-type]
            max_sequence_length=8,
            embedding_dim=16,
            num_heads=4,
            num_layers=2,
            feedforward_dim=64,
            dropout_rate=0.1,
        )


# ---------------------------------------------------------------------------
# test_decoder_only_transformer_rejects_invalid_max_sequence_length
#
# This test checks that max_sequence_length must be a positive integer.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad_max_sequence_length", [0, -1, "8"])
def test_decoder_only_transformer_rejects_invalid_max_sequence_length(
    bad_max_sequence_length: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="max_sequence_length must be a positive integer",
    ):
        DecoderOnlyTransformer(
            vocab_size=30,
            max_sequence_length=bad_max_sequence_length,  # type: ignore[arg-type]
            embedding_dim=16,
            num_heads=4,
            num_layers=2,
            feedforward_dim=64,
            dropout_rate=0.1,
        )


# ---------------------------------------------------------------------------
# test_decoder_only_transformer_rejects_invalid_embedding_dim
#
# This test checks that embedding_dim must be a positive integer.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad_embedding_dim", [0, -1, "16"])
def test_decoder_only_transformer_rejects_invalid_embedding_dim(
    bad_embedding_dim: int,
) -> None:
    with pytest.raises(ValueError, match="embedding_dim must be a positive integer"):
        DecoderOnlyTransformer(
            vocab_size=30,
            max_sequence_length=8,
            embedding_dim=bad_embedding_dim,  # type: ignore[arg-type]
            num_heads=4,
            num_layers=2,
            feedforward_dim=64,
            dropout_rate=0.1,
        )


# ---------------------------------------------------------------------------
# test_decoder_only_transformer_rejects_invalid_num_heads
#
# This test checks that num_heads must be a positive integer.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad_num_heads", [0, -1, "4"])
def test_decoder_only_transformer_rejects_invalid_num_heads(
    bad_num_heads: int,
) -> None:
    with pytest.raises(ValueError, match="num_heads must be a positive integer"):
        DecoderOnlyTransformer(
            vocab_size=30,
            max_sequence_length=8,
            embedding_dim=16,
            num_heads=bad_num_heads,  # type: ignore[arg-type]
            num_layers=2,
            feedforward_dim=64,
            dropout_rate=0.1,
        )


# ---------------------------------------------------------------------------
# test_decoder_only_transformer_rejects_invalid_num_layers
#
# This test checks that num_layers must be a positive integer.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad_num_layers", [0, -1, "2"])
def test_decoder_only_transformer_rejects_invalid_num_layers(
    bad_num_layers: int,
) -> None:
    with pytest.raises(ValueError, match="num_layers must be a positive integer"):
        DecoderOnlyTransformer(
            vocab_size=30,
            max_sequence_length=8,
            embedding_dim=16,
            num_heads=4,
            num_layers=bad_num_layers,  # type: ignore[arg-type]
            feedforward_dim=64,
            dropout_rate=0.1,
        )


# ---------------------------------------------------------------------------
# test_decoder_only_transformer_rejects_invalid_feedforward_dim
#
# This test checks that feedforward_dim must be a positive integer.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad_feedforward_dim", [0, -1, "64"])
def test_decoder_only_transformer_rejects_invalid_feedforward_dim(
    bad_feedforward_dim: int,
) -> None:
    with pytest.raises(ValueError, match="feedforward_dim must be a positive integer"):
        DecoderOnlyTransformer(
            vocab_size=30,
            max_sequence_length=8,
            embedding_dim=16,
            num_heads=4,
            num_layers=2,
            feedforward_dim=bad_feedforward_dim,  # type: ignore[arg-type]
            dropout_rate=0.1,
        )


# ---------------------------------------------------------------------------
# test_decoder_only_transformer_rejects_invalid_dropout_rate
#
# This test checks that dropout_rate must stay in the valid range.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad_dropout_rate", [-0.1, 1.0, 1.5, "0.1"])
def test_decoder_only_transformer_rejects_invalid_dropout_rate(
    bad_dropout_rate: float,
) -> None:
    with pytest.raises(ValueError, match="dropout_rate must be between 0.0 and 1.0"):
        DecoderOnlyTransformer(
            vocab_size=30,
            max_sequence_length=8,
            embedding_dim=16,
            num_heads=4,
            num_layers=2,
            feedforward_dim=64,
            dropout_rate=bad_dropout_rate,  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# test_decoder_only_transformer_rejects_non_tensor_input
#
# This test checks that token_ids must be a torch tensor.
# ---------------------------------------------------------------------------
def test_decoder_only_transformer_rejects_non_tensor_input() -> None:
    model = DecoderOnlyTransformer(
        vocab_size=30,
        max_sequence_length=8,
        embedding_dim=16,
        num_heads=4,
        num_layers=2,
        feedforward_dim=64,
        dropout_rate=0.1,
    )

    with pytest.raises(TypeError, match="token_ids must be a torch.Tensor"):
        model([[1, 2, 3]])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# test_decoder_only_transformer_rejects_non_2d_input
#
# This test checks that token_ids must be a 2D tensor.
# ---------------------------------------------------------------------------
def test_decoder_only_transformer_rejects_non_2d_input() -> None:
    model = DecoderOnlyTransformer(
        vocab_size=30,
        max_sequence_length=8,
        embedding_dim=16,
        num_heads=4,
        num_layers=2,
        feedforward_dim=64,
        dropout_rate=0.1,
    )
    token_ids = torch.tensor([1, 2, 3], dtype=torch.long)

    with pytest.raises(ValueError, match="token_ids must have shape"):
        model(token_ids)


# ---------------------------------------------------------------------------
# test_decoder_only_transformer_rejects_sequence_length_above_max
#
# This test checks that input sequence length cannot exceed the configured
# maximum sequence length.
# ---------------------------------------------------------------------------
def test_decoder_only_transformer_rejects_sequence_length_above_max() -> None:
    model = DecoderOnlyTransformer(
        vocab_size=30,
        max_sequence_length=4,
        embedding_dim=16,
        num_heads=4,
        num_layers=2,
        feedforward_dim=64,
        dropout_rate=0.1,
    )
    token_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)

    with pytest.raises(ValueError, match="cannot exceed max_sequence_length"):
        model(token_ids)


# ---------------------------------------------------------------------------
# test_decoder_only_transformer_returns_finite_values
#
# This test checks that the logits do not contain NaN or infinite values for a
# normal input tensor.
# ---------------------------------------------------------------------------
def test_decoder_only_transformer_returns_finite_values() -> None:
    model = DecoderOnlyTransformer(
        vocab_size=30,
        max_sequence_length=8,
        embedding_dim=16,
        num_heads=4,
        num_layers=2,
        feedforward_dim=64,
        dropout_rate=0.0,
    )
    token_ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)

    logits = model(token_ids)

    assert torch.isfinite(logits).all()


# ---------------------------------------------------------------------------
# test_decoder_only_transformer_preserves_device
#
# This test checks that the logits stay on the same device as the input.
# ---------------------------------------------------------------------------
def test_decoder_only_transformer_preserves_device() -> None:
    model = DecoderOnlyTransformer(
        vocab_size=30,
        max_sequence_length=8,
        embedding_dim=16,
        num_heads=4,
        num_layers=2,
        feedforward_dim=64,
        dropout_rate=0.0,
    )
    token_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

    logits = model(token_ids)

    assert logits.device == token_ids.device


# ---------------------------------------------------------------------------
# test_decoder_only_transformer_supports_single_token_sequence
#
# This test checks that a sequence length of one still works.
# ---------------------------------------------------------------------------
def test_decoder_only_transformer_supports_single_token_sequence() -> None:
    model = DecoderOnlyTransformer(
        vocab_size=30,
        max_sequence_length=8,
        embedding_dim=16,
        num_heads=4,
        num_layers=2,
        feedforward_dim=64,
        dropout_rate=0.0,
    )
    token_ids = torch.tensor([[1], [2]], dtype=torch.long)

    logits = model(token_ids)

    assert logits.shape == (2, 1, 30)
    assert torch.isfinite(logits).all()
