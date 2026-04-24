from __future__ import annotations

import pytest
import torch

from tiny_transformer.inference.sampling.greedy import greedy_sample_next_token


# ===========================================================================
# greedy_sample_next_token tests
# ===========================================================================


# ---------------------------------------------------------------------------
# test_greedy_sample_next_token_returns_expected_token_ids
#
# This test checks that greedy sampling picks the highest-logit token from
# each batch row.
# ---------------------------------------------------------------------------
def test_greedy_sample_next_token_returns_expected_token_ids() -> None:
    logits = torch.tensor(
        [
            [0.1, 0.8, 0.2],
            [1.2, 0.4, 0.3],
            [0.0, 0.2, 2.5],
        ]
    )

    next_token_ids = greedy_sample_next_token(logits)

    assert torch.equal(next_token_ids, torch.tensor([1, 0, 2]))


# ---------------------------------------------------------------------------
# test_greedy_sample_next_token_returns_1d_tensor
#
# This test checks that one token ID is returned per batch row.
# ---------------------------------------------------------------------------
def test_greedy_sample_next_token_returns_1d_tensor() -> None:
    logits = torch.randn(4, 6)

    next_token_ids = greedy_sample_next_token(logits)

    assert next_token_ids.dim() == 1
    assert next_token_ids.shape == (4,)


# ---------------------------------------------------------------------------
# test_greedy_sample_next_token_rejects_non_tensor_logits
#
# This test checks that logits must be a tensor.
# ---------------------------------------------------------------------------
def test_greedy_sample_next_token_rejects_non_tensor_logits() -> None:
    with pytest.raises(TypeError, match="logits must be a torch.Tensor"):
        greedy_sample_next_token([[0.1, 0.2, 0.3]])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# test_greedy_sample_next_token_rejects_non_2d_logits
#
# This test checks that logits must keep batch and vocab dimensions.
# ---------------------------------------------------------------------------
def test_greedy_sample_next_token_rejects_non_2d_logits() -> None:
    logits = torch.randn(3)

    with pytest.raises(
        ValueError,
        match="logits must have shape \\[batch_size, vocab_size\\]",
    ):
        greedy_sample_next_token(logits)


# ---------------------------------------------------------------------------
# test_greedy_sample_next_token_preserves_device
#
# This test checks that returned token IDs stay on the same device.
# ---------------------------------------------------------------------------
def test_greedy_sample_next_token_preserves_device() -> None:
    logits = torch.randn(2, 5)

    next_token_ids = greedy_sample_next_token(logits)

    assert next_token_ids.device == logits.device
