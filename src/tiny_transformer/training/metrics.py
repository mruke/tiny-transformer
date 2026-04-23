from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# _validate_loss_value
#
# This function checks that one loss value is a real number.
# Loss values must be finite so average reporting stays meaningful.
# ---------------------------------------------------------------------------
def _validate_loss_value(loss_value: float) -> None:
    if not isinstance(loss_value, (int, float)):
        raise TypeError("loss_value must be a real number.")

    if loss_value != loss_value:
        raise ValueError("loss_value must not be NaN.")

    if loss_value in (float("inf"), float("-inf")):
        raise ValueError("loss_value must be finite.")


# ---------------------------------------------------------------------------
# _validate_batch_count
#
# This function checks that batch count is a positive integer.
# ---------------------------------------------------------------------------
def _validate_batch_count(batch_count: int) -> None:
    if not isinstance(batch_count, int):
        raise TypeError("batch_count must be an integer.")

    if batch_count <= 0:
        raise ValueError("batch_count must be greater than zero.")


# ---------------------------------------------------------------------------
# calculate_average_loss
#
# This function returns the average loss across all processed batches.
# ---------------------------------------------------------------------------
def calculate_average_loss(total_loss: float, batch_count: int) -> float:
    _validate_loss_value(total_loss)
    _validate_batch_count(batch_count)

    return float(total_loss / batch_count)


# ---------------------------------------------------------------------------
# LossTracker
#
# LossTracker stores running loss totals for batch-based training or
# validation loops.
# ---------------------------------------------------------------------------
@dataclass
class LossTracker:
    total_loss: float = 0.0
    batch_count: int = 0

    # -----------------------------------------------------------------------
    # LossTracker.update
    #
    # This method adds one batch loss to the running totals.
    # -----------------------------------------------------------------------
    def update(self, loss_value: float) -> None:
        _validate_loss_value(loss_value)

        self.total_loss += float(loss_value)
        self.batch_count += 1

    # -----------------------------------------------------------------------
    # LossTracker.average_loss
    #
    # This method returns the current average loss.
    # -----------------------------------------------------------------------
    def average_loss(self) -> float:
        return calculate_average_loss(
            total_loss=self.total_loss,
            batch_count=self.batch_count,
        )
