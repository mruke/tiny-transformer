from __future__ import annotations

import pytest
import torch

from tiny_transformer.training.losses import compute_next_token_loss


# ===========================================================================
# compute_next_token_loss tests
# ===========================================================================


# ---------------------------------------------------------------------------
# test_compute_next_token_loss_returns_scalar_tensor
#
# This test checks that the loss function returns one scalar value.
# ---------------------------------------------------------------------------
def test_compute_next_token_loss_returns_scalar_tensor() -> None:
    logits = torch.randn(2, 3, 5, requires_grad=True)
    targets = torch.tensor([[1, 2, 3], [0, 4, 1]], dtype=torch.long)

    loss = compute_next_token_loss(logits, targets)

    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0


# ---------------------------------------------------------------------------
# test_compute_next_token_loss_returns_finite_value
#
# This test checks that valid inputs produce a finite loss value.
# ---------------------------------------------------------------------------
def test_compute_next_token_loss_returns_finite_value() -> None:
    logits = torch.randn(2, 4, 6, requires_grad=True)
    targets = torch.tensor([[1, 2, 3, 4], [0, 5, 1, 2]], dtype=torch.long)

    loss = compute_next_token_loss(logits, targets)

    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# test_compute_next_token_loss_supports_backward_pass
#
# This test checks that the computed loss can backpropagate gradients.
# ---------------------------------------------------------------------------
def test_compute_next_token_loss_supports_backward_pass() -> None:
    logits = torch.randn(2, 3, 4, requires_grad=True)
    targets = torch.tensor([[1, 2, 3], [0, 1, 2]], dtype=torch.long)

    loss = compute_next_token_loss(logits, targets)
    loss.backward()

    assert logits.grad is not None
    assert logits.grad.shape == logits.shape
    assert torch.isfinite(logits.grad).all()


# ---------------------------------------------------------------------------
# test_compute_next_token_loss_rejects_non_tensor_logits
#
# This test checks that logits must be a tensor.
# ---------------------------------------------------------------------------
def test_compute_next_token_loss_rejects_non_tensor_logits() -> None:
    targets = torch.tensor([[1, 2], [0, 1]], dtype=torch.long)

    with pytest.raises(TypeError, match="logits must be a torch.Tensor"):
        compute_next_token_loss([[0.1, 0.2]], targets)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# test_compute_next_token_loss_rejects_non_tensor_targets
#
# This test checks that targets must be a tensor.
# ---------------------------------------------------------------------------
def test_compute_next_token_loss_rejects_non_tensor_targets() -> None:
    logits = torch.randn(2, 2, 3)

    with pytest.raises(TypeError, match="targets must be a torch.Tensor"):
        compute_next_token_loss(logits, [[1, 2], [0, 1]])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# test_compute_next_token_loss_rejects_non_3d_logits
#
# This test checks that logits must keep batch, sequence, and vocab dimensions.
# ---------------------------------------------------------------------------
def test_compute_next_token_loss_rejects_non_3d_logits() -> None:
    logits = torch.randn(2, 3)
    targets = torch.tensor([[1, 2, 0], [0, 1, 2]], dtype=torch.long)

    with pytest.raises(
        ValueError,
        match="logits must have shape \\[batch_size, sequence_length, vocab_size\\]",
    ):
        compute_next_token_loss(logits, targets)


# ---------------------------------------------------------------------------
# test_compute_next_token_loss_rejects_non_2d_targets
#
# This test checks that targets must keep batch and sequence dimensions.
# ---------------------------------------------------------------------------
def test_compute_next_token_loss_rejects_non_2d_targets() -> None:
    logits = torch.randn(2, 3, 5)
    targets = torch.tensor([1, 2, 3], dtype=torch.long)

    with pytest.raises(
        ValueError,
        match="targets must have shape \\[batch_size, sequence_length\\]",
    ):
        compute_next_token_loss(logits, targets)


# ---------------------------------------------------------------------------
# test_compute_next_token_loss_rejects_non_long_targets
#
# This test checks that targets must use torch.long dtype.
# ---------------------------------------------------------------------------
def test_compute_next_token_loss_rejects_non_long_targets() -> None:
    logits = torch.randn(2, 3, 5)
    targets = torch.tensor([[1, 2, 3], [0, 1, 2]], dtype=torch.int32)

    with pytest.raises(TypeError, match="targets must use torch.long dtype"):
        compute_next_token_loss(logits, targets)


# ---------------------------------------------------------------------------
# test_compute_next_token_loss_rejects_batch_size_mismatch
#
# This test checks that logits and targets must agree on batch size.
# ---------------------------------------------------------------------------
def test_compute_next_token_loss_rejects_batch_size_mismatch() -> None:
    logits = torch.randn(2, 3, 5)
    targets = torch.tensor([[1, 2, 3]], dtype=torch.long)

    with pytest.raises(ValueError, match="same batch_size"):
        compute_next_token_loss(logits, targets)


# ---------------------------------------------------------------------------
# test_compute_next_token_loss_rejects_sequence_length_mismatch
#
# This test checks that logits and targets must agree on sequence length.
# ---------------------------------------------------------------------------
def test_compute_next_token_loss_rejects_sequence_length_mismatch() -> None:
    logits = torch.randn(2, 3, 5)
    targets = torch.tensor([[1, 2], [0, 1]], dtype=torch.long)

    with pytest.raises(ValueError, match="same sequence_length"):
        compute_next_token_loss(logits, targets)
