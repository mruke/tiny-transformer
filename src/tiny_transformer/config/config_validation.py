from __future__ import annotations

from typing import Any

from tiny_transformer.config.config_exceptions import ConfigError


# ---------------------------------------------------------------------------
# _require_mapping
#
# This function checks that a config section is a dictionary.
# Each top-level YAML section should be a mapping.
# ---------------------------------------------------------------------------
def _require_mapping(value: Any, section_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"Configuration section '{section_name}' must be a mapping.")
    return value


# ---------------------------------------------------------------------------
# _require_non_empty_string
#
# This function checks that a field is a non-empty string.
# Blank strings are not allowed for required text values.
# ---------------------------------------------------------------------------
def _require_non_empty_string(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"'{field_name}' must be a non-empty string.")


# ---------------------------------------------------------------------------
# _require_positive_int
#
# This function checks that a field is a positive integer.
# Positive means greater than zero.
# ---------------------------------------------------------------------------
def _require_positive_int(value: int, field_name: str) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ConfigError(f"'{field_name}' must be a positive integer.")


# ---------------------------------------------------------------------------
# _require_non_negative_int
#
# This function checks that a field is a non-negative integer.
# Non-negative means zero or greater.
# ---------------------------------------------------------------------------
def _require_non_negative_int(value: int, field_name: str) -> None:
    if not isinstance(value, int) or value < 0:
        raise ConfigError(f"'{field_name}' must be a non-negative integer.")


# ---------------------------------------------------------------------------
# _require_positive_number
#
# This function checks that a field is a positive number.
# This works for both integers and floats.
# ---------------------------------------------------------------------------
def _require_positive_number(value: float, field_name: str) -> None:
    if not isinstance(value, (int, float)) or value <= 0:
        raise ConfigError(f"'{field_name}' must be a positive number.")


# ---------------------------------------------------------------------------
# _require_non_negative_number
#
# This function checks that a field is a non-negative number.
# This works for both integers and floats.
# ---------------------------------------------------------------------------
def _require_non_negative_number(value: float, field_name: str) -> None:
    if not isinstance(value, (int, float)) or value < 0:
        raise ConfigError(f"'{field_name}' must be a non-negative number.")
