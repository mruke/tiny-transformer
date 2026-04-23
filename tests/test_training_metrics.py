from __future__ import annotations

import pytest

from tiny_transformer.training.metrics import LossTracker, calculate_average_loss


# ===========================================================================
# calculate_average_loss tests
# ===========================================================================


# ---------------------------------------------------------------------------
# test_calculate_average_loss_returns_expected_average
#
# This test checks that average loss is computed correctly.
# ---------------------------------------------------------------------------
def test_calculate_average_loss_returns_expected_average() -> None:
    average_loss = calculate_average_loss(total_loss=9.0, batch_count=3)

    assert average_loss == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# test_calculate_average_loss_rejects_zero_batch_count
#
# This test checks that batch count must be greater than zero.
# ---------------------------------------------------------------------------
def test_calculate_average_loss_rejects_zero_batch_count() -> None:
    with pytest.raises(ValueError, match="batch_count must be greater than zero"):
        calculate_average_loss(total_loss=4.0, batch_count=0)


# ---------------------------------------------------------------------------
# test_calculate_average_loss_rejects_nan_total_loss
#
# This test checks that total loss must not be NaN.
# ---------------------------------------------------------------------------
def test_calculate_average_loss_rejects_nan_total_loss() -> None:
    with pytest.raises(ValueError, match="loss_value must not be NaN"):
        calculate_average_loss(total_loss=float("nan"), batch_count=2)


# ---------------------------------------------------------------------------
# test_calculate_average_loss_rejects_infinite_total_loss
#
# This test checks that total loss must be finite.
# ---------------------------------------------------------------------------
def test_calculate_average_loss_rejects_infinite_total_loss() -> None:
    with pytest.raises(ValueError, match="loss_value must be finite"):
        calculate_average_loss(total_loss=float("inf"), batch_count=2)


# ===========================================================================
# LossTracker tests
# ===========================================================================


# ---------------------------------------------------------------------------
# test_loss_tracker_update_accumulates_loss_and_batch_count
#
# This test checks that tracker updates increase the running totals.
# ---------------------------------------------------------------------------
def test_loss_tracker_update_accumulates_loss_and_batch_count() -> None:
    tracker = LossTracker()

    tracker.update(2.0)
    tracker.update(4.0)

    assert tracker.total_loss == pytest.approx(6.0)
    assert tracker.batch_count == 2


# ---------------------------------------------------------------------------
# test_loss_tracker_average_loss_returns_expected_value
#
# This test checks that tracker average loss is computed correctly.
# ---------------------------------------------------------------------------
def test_loss_tracker_average_loss_returns_expected_value() -> None:
    tracker = LossTracker()

    tracker.update(1.5)
    tracker.update(2.5)
    tracker.update(3.0)

    assert tracker.average_loss() == pytest.approx(7.0 / 3.0)


# ---------------------------------------------------------------------------
# test_loss_tracker_average_loss_rejects_empty_tracker
#
# This test checks that average loss cannot be read before any updates.
# ---------------------------------------------------------------------------
def test_loss_tracker_average_loss_rejects_empty_tracker() -> None:
    tracker = LossTracker()

    with pytest.raises(ValueError, match="batch_count must be greater than zero"):
        tracker.average_loss()


# ---------------------------------------------------------------------------
# test_loss_tracker_update_rejects_non_numeric_loss
#
# This test checks that loss updates must use numeric values.
# ---------------------------------------------------------------------------
def test_loss_tracker_update_rejects_non_numeric_loss() -> None:
    tracker = LossTracker()

    with pytest.raises(TypeError, match="loss_value must be a real number"):
        tracker.update("bad-loss")  # type: ignore[arg-type]
