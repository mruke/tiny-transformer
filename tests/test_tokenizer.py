from __future__ import annotations

import pytest

from tiny_transformer.data.tokenizer import CharacterTokenizer


# ---------------------------------------------------------------------------
# test_from_text_builds_deterministic_vocabulary
#
# This test checks that the same input text always builds the same vocabulary.
# ---------------------------------------------------------------------------
def test_from_text_builds_deterministic_vocabulary() -> None:
    text = "banana"

    tokenizer_a = CharacterTokenizer.from_text(text)
    tokenizer_b = CharacterTokenizer.from_text(text)

    assert tokenizer_a.char_to_id == tokenizer_b.char_to_id
    assert tokenizer_a.id_to_char == tokenizer_b.id_to_char


# ---------------------------------------------------------------------------
# test_vocab_size_matches_unique_character_count
#
# This test checks that vocab size matches the number of unique characters.
# ---------------------------------------------------------------------------
def test_vocab_size_matches_unique_character_count() -> None:
    text = "banana"

    tokenizer = CharacterTokenizer.from_text(text)

    assert tokenizer.vocab_size == 3


# ---------------------------------------------------------------------------
# test_encode_returns_expected_token_ids
#
# This test checks that encoding follows the sorted vocabulary order.
# ---------------------------------------------------------------------------
def test_encode_returns_expected_token_ids() -> None:
    text = "banana"
    tokenizer = CharacterTokenizer.from_text(text)

    # Sorted unique characters are: ["a", "b", "n"]
    assert tokenizer.char_to_id == {"a": 0, "b": 1, "n": 2}
    assert tokenizer.encode("banana") == [1, 0, 2, 0, 2, 0]


# ---------------------------------------------------------------------------
# test_decode_returns_expected_text
#
# This test checks that decoding restores text from token IDs.
# ---------------------------------------------------------------------------
def test_decode_returns_expected_text() -> None:
    tokenizer = CharacterTokenizer.from_text("banana")

    decoded_text = tokenizer.decode([1, 0, 2, 0, 2, 0])

    assert decoded_text == "banana"


# ---------------------------------------------------------------------------
# test_encode_decode_round_trip_returns_original_text
#
# This test checks that encoding and then decoding returns the same text.
# ---------------------------------------------------------------------------
def test_encode_decode_round_trip_returns_original_text() -> None:
    original_text = "hello world"
    tokenizer = CharacterTokenizer.from_text(original_text)

    encoded = tokenizer.encode(original_text)
    decoded = tokenizer.decode(encoded)

    assert decoded == original_text


# ---------------------------------------------------------------------------
# test_encode_unknown_character_raises_error
#
# This test checks that encoding a character outside the vocabulary fails.
# ---------------------------------------------------------------------------
def test_encode_unknown_character_raises_error() -> None:
    tokenizer = CharacterTokenizer.from_text("abc")

    with pytest.raises(ValueError, match="is not in the tokenizer vocabulary"):
        tokenizer.encode("abd")


# ---------------------------------------------------------------------------
# test_decode_unknown_token_id_raises_error
#
# This test checks that decoding an unknown token ID fails.
# ---------------------------------------------------------------------------
def test_decode_unknown_token_id_raises_error() -> None:
    tokenizer = CharacterTokenizer.from_text("abc")

    with pytest.raises(ValueError, match="is not in the tokenizer vocabulary"):
        tokenizer.decode([0, 1, 99])


# ---------------------------------------------------------------------------
# test_from_text_rejects_empty_text
#
# This test checks that building a tokenizer from empty text fails.
# ---------------------------------------------------------------------------
def test_from_text_rejects_empty_text() -> None:
    with pytest.raises(ValueError, match="must be non-empty"):
        CharacterTokenizer.from_text("")


# ---------------------------------------------------------------------------
# test_decode_rejects_non_integer_token_ids
#
# This test checks that token IDs must be integers.
# ---------------------------------------------------------------------------
def test_decode_rejects_non_integer_token_ids() -> None:
    tokenizer = CharacterTokenizer.from_text("abc")

    with pytest.raises(TypeError, match="must be an integer"):
        tokenizer.decode([0, "1"])  # type: ignore[list-item]
