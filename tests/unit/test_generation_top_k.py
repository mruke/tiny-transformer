from __future__ import annotations

import pytest
import torch

from tiny_transformer.inference.sampling.top_k import apply_top_k_filter


# ===========================================================================
# apply_top_k_filter tests
# ===========================================================================


# ---------------------------------------------------------------------------
# test_apply_top_k_filter_with_none_returns_original_logits
#
# This test checks that top-k filtering is skipped when top_k is None.
# ---------------------------------------------------------------------------
def test_apply_top_k_filter_with_none_returns_original_logits() -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0]])

    filtered_logits = apply_top_k_filter(logits=logits, top_k=None)

    assert torch.equal(filtered_logits, logits)


# ---------------------------------------------------------------------------
# test_apply_top_k_filter_keeps_only_top_k_logits
#
# This test checks that only the top-k logits remain finite.
# ---------------------------------------------------------------------------
def test_apply_top_k_filter_keeps_only_top_k_logits() -> None:
    logits = torch.tensor([[10.0, 9.0, -10.0, -10.0]])

    filtered_logits = apply_top_k_filter(logits=logits, top_k=2)

    assert torch.isfinite(filtered_logits[0, 0])
    assert torch.isfinite(filtered_logits[0, 1])
    assert filtered_logits[0, 2].item() == float("-inf")
    assert filtered_logits[0, 3].item() == float("-inf")


# ---------------------------------------------------------------------------
# test_apply_top_k_filter_rejects_non_tensor_logits
#
# This test checks that logits must be a tensor.
# ---------------------------------------------------------------------------
def test_apply_top_k_filter_rejects_non_tensor_logits() -> None:
    with pytest.raises(TypeError, match="logits must be a torch.Tensor"):
        apply_top_k_filter(logits=[[0.1, 0.2, 0.3]], top_k=2)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# test_apply_top_k_filter_rejects_non_2d_logits
#
# This test checks that logits must keep batch and vocab dimensions.
# ---------------------------------------------------------------------------
def test_apply_top_k_filter_rejects_non_2d_logits() -> None:
    logits = torch.randn(3)

    with pytest.raises(
        ValueError,
        match="logits must have shape \\[batch_size, vocab_size\\]",
    ):
        apply_top_k_filter(logits=logits, top_k=2)


# ---------------------------------------------------------------------------
# test_apply_top_k_filter_rejects_non_integer_top_k
#
# This test checks that top_k must be an integer or None.
# ---------------------------------------------------------------------------
def test_apply_top_k_filter_rejects_non_integer_top_k() -> None:
    logits = torch.randn(2, 5)

    with pytest.raises(TypeError, match="top_k must be an integer or None"):
        apply_top_k_filter(logits=logits, top_k=2.5)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# test_apply_top_k_filter_rejects_non_positive_top_k
#
# This test checks that top_k must be greater than zero.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("top_k", [0, -1])
def test_apply_top_k_filter_rejects_non_positive_top_k(top_k: int) -> None:
    logits = torch.randn(2, 5)

    with pytest.raises(ValueError, match="top_k must be greater than zero"):
        apply_top_k_filter(logits=logits, top_k=top_k)


# ---------------------------------------------------------------------------
# test_apply_top_k_filter_rejects_top_k_above_vocab_size
#
# This test checks that top_k must not exceed vocab size.
# ---------------------------------------------------------------------------
def test_apply_top_k_filter_rejects_top_k_above_vocab_size() -> None:
    logits = torch.randn(2, 5)

    with pytest.raises(ValueError, match="top_k must not exceed vocab_size"):
        apply_top_k_filter(logits=logits, top_k=6)
