from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from tiny_transformer.config.config_exceptions import ConfigError
from tiny_transformer.config.config_schema import (
    AppConfig,
    CheckpointingConfig,
    DataConfig,
    GenerationConfig,
    LoggingConfig,
    ModelConfig,
    ProjectConfig,
    TrainingConfig,
)
from tiny_transformer.config.config_validation import _require_mapping


# ---------------------------------------------------------------------------
# _build_app_config
#
# This function builds a full AppConfig from a raw dictionary.
# Each section is validated by its own dataclass.
# ---------------------------------------------------------------------------
def _build_app_config(raw_config: dict[str, Any]) -> AppConfig:
    project_section = _require_mapping(raw_config.get("project"), "project")
    data_section = _require_mapping(raw_config.get("data"), "data")
    model_section = _require_mapping(raw_config.get("model"), "model")
    training_section = _require_mapping(raw_config.get("training"), "training")
    generation_section = _require_mapping(raw_config.get("generation"), "generation")
    logging_section = _require_mapping(raw_config.get("logging"), "logging")
    checkpointing_section = _require_mapping(
        raw_config.get("checkpointing"),
        "checkpointing",
    )

    return AppConfig(
        project=ProjectConfig(**project_section),
        data=DataConfig(**data_section),
        model=ModelConfig(**model_section),
        training=TrainingConfig(**training_section),
        generation=GenerationConfig(**generation_section),
        logging=LoggingConfig(**logging_section),
        checkpointing=CheckpointingConfig(**checkpointing_section),
    )


# ---------------------------------------------------------------------------
# load_config
#
# load_config reads a YAML file, checks that it has the right top-level shape,
# and returns a validated AppConfig object.
# ---------------------------------------------------------------------------
def load_config(config_path: str | Path) -> AppConfig:
    path = Path(config_path)

    if not path.exists():
        raise ConfigError(f"Config file does not exist: {path}")

    if not path.is_file():
        raise ConfigError(f"Config path is not a file: {path}")

    try:
        with path.open("r", encoding="utf-8") as config_file:
            raw_config = yaml.safe_load(config_file)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Failed to parse YAML config: {path}") from exc

    if raw_config is None:
        raise ConfigError(f"Config file is empty: {path}")

    if not isinstance(raw_config, dict):
        raise ConfigError(f"Top-level configuration must be a mapping: {path}")

    return _build_app_config(raw_config)
