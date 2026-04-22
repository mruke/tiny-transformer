from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# CharacterTokenizer
#
# CharacterTokenizer builds a character-level vocabulary from text.
# It can encode text into token IDs and decode token IDs back into text.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CharacterTokenizer:
    _char_to_id: dict[str, int] = field(repr=False)
    _id_to_char: dict[int, str] = field(repr=False)

    # -----------------------------------------------------------------------
    # CharacterTokenizer.__post_init__
    #
    # This method checks that the two mappings agree with each other.
    # This protects the tokenizer from bad internal state.
    # -----------------------------------------------------------------------
    def __post_init__(self) -> None:
        if not self._char_to_id:
            raise ValueError("Tokenizer vocabulary cannot be empty.")

        if len(self._char_to_id) != len(self._id_to_char):
            raise ValueError(
                "Character-to-id and id-to-character mappings must have the same size."
            )

        for char, token_id in self._char_to_id.items():
            if self._id_to_char.get(token_id) != char:
                raise ValueError(
                    "Character-to-id and id-to-character mappings must match."
                )

    # -----------------------------------------------------------------------
    # CharacterTokenizer.from_text
    #
    # This class method builds a tokenizer from raw text.
    # Characters are sorted so the vocabulary order stays deterministic.
    # -----------------------------------------------------------------------
    @classmethod
    def from_text(cls, text: str) -> "CharacterTokenizer":
        if not isinstance(text, str) or not text:
            raise ValueError("Text used to build the tokenizer must be non-empty.")

        unique_characters = sorted(set(text))

        char_to_id = {char: index for index, char in enumerate(unique_characters)}

        id_to_char = {index: char for char, index in char_to_id.items()}

        return cls(
            _char_to_id=char_to_id,
            _id_to_char=id_to_char,
        )

    # -----------------------------------------------------------------------
    # vocab_size
    #
    # This property returns the number of unique characters in the vocabulary.
    # -----------------------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return len(self._char_to_id)

    # -----------------------------------------------------------------------
    # char_to_id
    #
    # This property returns a copy of the character-to-id mapping.
    # A copy is returned so outside code cannot change tokenizer state.
    # -----------------------------------------------------------------------
    @property
    def char_to_id(self) -> dict[str, int]:
        return dict(self._char_to_id)

    # -----------------------------------------------------------------------
    # id_to_char
    #
    # This property returns a copy of the id-to-character mapping.
    # A copy is returned so outside code cannot change tokenizer state.
    # -----------------------------------------------------------------------
    @property
    def id_to_char(self) -> dict[int, str]:
        return dict(self._id_to_char)

    # -----------------------------------------------------------------------
    # encode
    #
    # This method turns text into a list of integer token IDs.
    # Every character must already be in the vocabulary.
    # -----------------------------------------------------------------------
    def encode(self, text: str) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("Text to encode must be a string.")

        token_ids: list[int] = []

        for char in text:
            if char not in self._char_to_id:
                raise ValueError(
                    f"Character {char!r} is not in the tokenizer vocabulary."
                )

            token_ids.append(self._char_to_id[char])

        return token_ids

    # -----------------------------------------------------------------------
    # decode
    #
    # This method turns a list of integer token IDs back into text.
    # Every token ID must already be in the vocabulary.
    # -----------------------------------------------------------------------
    def decode(self, token_ids: list[int]) -> str:
        if not isinstance(token_ids, list):
            raise TypeError("Token IDs to decode must be provided as a list.")

        characters: list[str] = []

        for token_id in token_ids:
            if not isinstance(token_id, int):
                raise TypeError("Each token ID must be an integer.")

            if token_id not in self._id_to_char:
                raise ValueError(
                    f"Token ID {token_id} is not in the tokenizer vocabulary."
                )

            characters.append(self._id_to_char[token_id])

        return "".join(characters)
