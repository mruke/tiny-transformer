from tiny_transformer.inference.sampling.greedy import greedy_sample_next_token
from tiny_transformer.inference.sampling.temperature import (
    temperature_sample_next_token,
)
from tiny_transformer.inference.sampling.top_k import apply_top_k_filter

__all__ = [
    "apply_top_k_filter",
    "greedy_sample_next_token",
    "temperature_sample_next_token",
]
