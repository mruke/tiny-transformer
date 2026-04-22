from __future__ import annotations

import math

import torch
from torch import nn

from tiny_transformer.model.masks import create_causal_attention_mask


# ---------------------------------------------------------------------------
# _validate_positive_int
#
# This function checks that a value is a positive integer.
# ---------------------------------------------------------------------------
def _validate_positive_int(value: int, field_name: str) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer.")


# ---------------------------------------------------------------------------
# _validate_dropout_rate
#
# This function checks that dropout stays in the valid range.
# ---------------------------------------------------------------------------
def _validate_dropout_rate(dropout_rate: float) -> None:
    if not isinstance(dropout_rate, (int, float)) or not (0.0 <= dropout_rate < 1.0):
        raise ValueError("dropout_rate must be between 0.0 and 1.0.")


# ---------------------------------------------------------------------------
# _validate_hidden_states
#
# This function checks that hidden states are a 3D tensor.
# Expected shape: [batch_size, sequence_length, embedding_dim]
# ---------------------------------------------------------------------------
def _validate_hidden_states(hidden_states: torch.Tensor) -> None:
    if not isinstance(hidden_states, torch.Tensor):
        raise TypeError("hidden_states must be a torch.Tensor.")

    if hidden_states.dim() != 3:
        raise ValueError(
            "hidden_states must have shape "
            "[batch_size, sequence_length, embedding_dim]."
        )


# ---------------------------------------------------------------------------
# _reshape_for_multi_head_attention
#
# This function reshapes projected hidden states into head format.
# Input shape:  [batch_size, sequence_length, embedding_dim]
# Output shape: [batch_size, num_heads, sequence_length, head_dim]
# ---------------------------------------------------------------------------
def _reshape_for_multi_head_attention(
    tensor: torch.Tensor,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    batch_size, sequence_length, embedding_dim = tensor.shape

    if embedding_dim != num_heads * head_dim:
        raise ValueError("Projected tensor shape does not match num_heads * head_dim.")

    tensor = tensor.view(batch_size, sequence_length, num_heads, head_dim)
    tensor = tensor.transpose(1, 2)

    return tensor


# ---------------------------------------------------------------------------
# _combine_attention_heads
#
# This function combines attention heads back into embedding space.
# Input shape:  [batch_size, num_heads, sequence_length, head_dim]
# Output shape: [batch_size, sequence_length, embedding_dim]
# ---------------------------------------------------------------------------
def _combine_attention_heads(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() != 4:
        raise ValueError(
            "Attention head tensor must have shape "
            "[batch_size, num_heads, sequence_length, head_dim]."
        )

    batch_size, num_heads, sequence_length, head_dim = tensor.shape

    tensor = tensor.transpose(1, 2).contiguous()
    tensor = tensor.view(batch_size, sequence_length, num_heads * head_dim)

    return tensor


# ---------------------------------------------------------------------------
# MultiHeadSelfAttention
#
# MultiHeadSelfAttention applies masked multi-head self-attention to hidden
# states. This module projects queries, keys, and values, applies a causal
# mask, computes attention, and combines the result back into model space.
# ---------------------------------------------------------------------------
class MultiHeadSelfAttention(nn.Module):
    # -----------------------------------------------------------------------
    # MultiHeadSelfAttention.__init__
    #
    # This method creates the projection layers and stores model dimensions.
    # -----------------------------------------------------------------------
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()

        _validate_positive_int(embedding_dim, "embedding_dim")
        _validate_positive_int(num_heads, "num_heads")
        _validate_dropout_rate(dropout_rate)

        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads.")

        self._embedding_dim = embedding_dim
        self._num_heads = num_heads
        self._head_dim = embedding_dim // num_heads

        self._query_projection = nn.Linear(embedding_dim, embedding_dim)
        self._key_projection = nn.Linear(embedding_dim, embedding_dim)
        self._value_projection = nn.Linear(embedding_dim, embedding_dim)
        self._output_projection = nn.Linear(embedding_dim, embedding_dim)

        self._attention_dropout = nn.Dropout(dropout_rate)

    # -----------------------------------------------------------------------
    # _validate_input_embedding_dim
    #
    # This method checks that the incoming hidden state size matches the
    # configured embedding size for the module.
    # -----------------------------------------------------------------------
    def _validate_input_embedding_dim(self, hidden_states: torch.Tensor) -> None:
        _, _, embedding_dim = hidden_states.shape

        if embedding_dim != self._embedding_dim:
            raise ValueError(
                "hidden_states embedding dimension does not match "
                "the configured embedding_dim."
            )

    # -----------------------------------------------------------------------
    # _project_query_key_value
    #
    # This method projects hidden states into query, key, and value tensors.
    # -----------------------------------------------------------------------
    def _project_query_key_value(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query = self._query_projection(hidden_states)
        key = self._key_projection(hidden_states)
        value = self._value_projection(hidden_states)

        return query, key, value

    # -----------------------------------------------------------------------
    # _reshape_projected_tensors
    #
    # This method reshapes projected query, key, and value tensors into
    # multi-head attention format.
    # -----------------------------------------------------------------------
    def _reshape_projected_tensors(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query = _reshape_for_multi_head_attention(
            query,
            num_heads=self._num_heads,
            head_dim=self._head_dim,
        )
        key = _reshape_for_multi_head_attention(
            key,
            num_heads=self._num_heads,
            head_dim=self._head_dim,
        )
        value = _reshape_for_multi_head_attention(
            value,
            num_heads=self._num_heads,
            head_dim=self._head_dim,
        )

        return query, key, value

    # -----------------------------------------------------------------------
    # _compute_attention_scores
    #
    # This method computes scaled dot-product attention scores.
    # Output shape:
    # [batch_size, num_heads, sequence_length, sequence_length]
    # -----------------------------------------------------------------------
    def _compute_attention_scores(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> torch.Tensor:
        attention_scores = torch.matmul(query, key.transpose(-2, -1))
        attention_scores = attention_scores / math.sqrt(self._head_dim)

        return attention_scores

    # -----------------------------------------------------------------------
    # _apply_causal_mask
    #
    # This method blocks attention to future token positions by applying the
    # causal mask to attention scores.
    # -----------------------------------------------------------------------
    def _apply_causal_mask(
        self,
        attention_scores: torch.Tensor,
        sequence_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        causal_mask = create_causal_attention_mask(sequence_length)
        causal_mask = causal_mask.to(device)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        masked_scores = attention_scores.masked_fill(
            causal_mask,
            float("-inf"),
        )

        return masked_scores

    # -----------------------------------------------------------------------
    # _compute_attention_weights
    #
    # This method turns masked attention scores into attention weights.
    # -----------------------------------------------------------------------
    def _compute_attention_weights(
        self,
        attention_scores: torch.Tensor,
    ) -> torch.Tensor:
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self._attention_dropout(attention_weights)

        return attention_weights

    # -----------------------------------------------------------------------
    # _apply_attention_to_values
    #
    # This method applies attention weights to value vectors and combines the
    # attention heads back into embedding space.
    # -----------------------------------------------------------------------
    def _apply_attention_to_values(
        self,
        attention_weights: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        attention_output = torch.matmul(attention_weights, value)
        attention_output = _combine_attention_heads(attention_output)

        return attention_output

    # -----------------------------------------------------------------------
    # forward
    #
    # This method applies masked multi-head self-attention.
    # Input shape:  [batch_size, sequence_length, embedding_dim]
    # Output shape: [batch_size, sequence_length, embedding_dim]
    # -----------------------------------------------------------------------
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        _validate_hidden_states(hidden_states)
        self._validate_input_embedding_dim(hidden_states)

        _, sequence_length, _ = hidden_states.shape

        query, key, value = self._project_query_key_value(hidden_states)
        query, key, value = self._reshape_projected_tensors(query, key, value)

        attention_scores = self._compute_attention_scores(query, key)
        attention_scores = self._apply_causal_mask(
            attention_scores=attention_scores,
            sequence_length=sequence_length,
            device=hidden_states.device,
        )

        attention_weights = self._compute_attention_weights(attention_scores)
        attention_output = self._apply_attention_to_values(
            attention_weights,
            value,
        )

        output = self._output_projection(attention_output)

        return output
