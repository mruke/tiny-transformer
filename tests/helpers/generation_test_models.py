from __future__ import annotations

import torch
from torch import nn


# ---------------------------------------------------------------------------
# GreedyTestModel
#
# This test model returns fixed logits so greedy generation behavior stays
# deterministic.
# ---------------------------------------------------------------------------
class GreedyTestModel(nn.Module):
    # -----------------------------------------------------------------------
    # GreedyTestModel.__init__
    #
    # This method stores the vocab size and fixed next-token choice.
    # -----------------------------------------------------------------------
    def __init__(
        self,
        vocab_size: int,
        next_token_id: int,
    ) -> None:
        super().__init__()
        self._vocab_size = vocab_size
        self._next_token_id = next_token_id

    # -----------------------------------------------------------------------
    # GreedyTestModel.forward
    #
    # This method returns logits that always favor the same next token.
    # -----------------------------------------------------------------------
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length = input_ids.shape
        logits = torch.zeros(
            batch_size,
            sequence_length,
            self._vocab_size,
            dtype=torch.float32,
            device=input_ids.device,
        )
        logits[:, :, self._next_token_id] = 1.0

        return logits


# ---------------------------------------------------------------------------
# TemperatureTestModel
#
# This test model returns fixed logits with two strong token choices so
# temperature sampling and top-k behavior can be exercised predictably.
# ---------------------------------------------------------------------------
class TemperatureTestModel(nn.Module):
    # -----------------------------------------------------------------------
    # TemperatureTestModel.__init__
    #
    # This method stores the vocab size.
    # -----------------------------------------------------------------------
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self._vocab_size = vocab_size

    # -----------------------------------------------------------------------
    # TemperatureTestModel.forward
    #
    # This method returns logits with token 1 and token 2 as the strongest
    # choices at every step.
    # -----------------------------------------------------------------------
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length = input_ids.shape
        logits = torch.full(
            (batch_size, sequence_length, self._vocab_size),
            fill_value=-10.0,
            dtype=torch.float32,
            device=input_ids.device,
        )
        logits[:, :, 1] = 10.0
        logits[:, :, 2] = 9.0

        return logits


# ---------------------------------------------------------------------------
# ContextWindowTrackingModel
#
# This test model records the sequence lengths it receives so context-window
# trimming can be verified.
# ---------------------------------------------------------------------------
class ContextWindowTrackingModel(nn.Module):
    # -----------------------------------------------------------------------
    # ContextWindowTrackingModel.__init__
    #
    # This method stores vocab size, next-token choice, and max sequence
    # length.
    # -----------------------------------------------------------------------
    def __init__(
        self,
        vocab_size: int,
        next_token_id: int,
        max_sequence_length: int,
    ) -> None:
        super().__init__()
        self._vocab_size = vocab_size
        self._next_token_id = next_token_id
        self._max_sequence_length = max_sequence_length
        self.seen_sequence_lengths: list[int] = []

    # -----------------------------------------------------------------------
    # max_sequence_length
    #
    # This property returns the configured maximum supported sequence length.
    # -----------------------------------------------------------------------
    @property
    def max_sequence_length(self) -> int:
        return self._max_sequence_length

    # -----------------------------------------------------------------------
    # ContextWindowTrackingModel.forward
    #
    # This method records seen sequence lengths and returns fixed logits.
    # -----------------------------------------------------------------------
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length = input_ids.shape
        self.seen_sequence_lengths.append(sequence_length)

        logits = torch.zeros(
            batch_size,
            sequence_length,
            self._vocab_size,
            dtype=torch.float32,
            device=input_ids.device,
        )
        logits[:, :, self._next_token_id] = 1.0

        return logits


# ---------------------------------------------------------------------------
# ValidationTestModel
#
# This test model returns placeholder logits for generator validation tests.
# ---------------------------------------------------------------------------
class ValidationTestModel(nn.Module):
    # -----------------------------------------------------------------------
    # ValidationTestModel.forward
    #
    # This method returns fixed placeholder logits.
    # -----------------------------------------------------------------------
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length = input_ids.shape

        return torch.zeros(
            batch_size,
            sequence_length,
            4,
            dtype=torch.float32,
            device=input_ids.device,
        )
