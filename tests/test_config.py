from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tiny_transformer.config import AppConfig, ConfigError, load_config


# ---------------------------------------------------------------------------
# _repo_root
#
# This helper finds the root of the repository from the test file location.
# That makes it easy to load the real config files from tests.
# ---------------------------------------------------------------------------
def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# _write_yaml
#
# This helper writes a small YAML file into the temporary test directory.
# It is used for invalid test cases and small custom config cases.
# ---------------------------------------------------------------------------
def _write_yaml(tmp_path: Path, filename: str, content: dict) -> Path:
    file_path = tmp_path / filename
    file_path.write_text(yaml.safe_dump(content), encoding="utf-8")
    return file_path


# ---------------------------------------------------------------------------
# _valid_config_dict
#
# This helper returns a complete valid config dictionary.
# Test cases can copy this and change one part at a time.
# ---------------------------------------------------------------------------
def _valid_config_dict() -> dict:
    return {
        "project": {
            "name": "tiny-transformer",
            "version": "0.1.0",
            "seed": 42,
        },
        "data": {
            "dataset_path": "data/input.txt",
            "train_split_ratio": 0.9,
            "context_window": 128,
            "encoding": "utf-8",
        },
        "model": {
            "vocab_size": None,
            "max_sequence_length": 128,
            "embedding_dim": 128,
            "num_heads": 4,
            "num_layers": 2,
            "feedforward_dim": 512,
            "dropout_rate": 0.1,
        },
        "training": {
            "batch_size": 32,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "max_epochs": 10,
            "device": "cpu",
            "eval_interval": 100,
            "log_interval": 10,
        },
        "generation": {
            "max_new_tokens": 100,
            "temperature": 1.0,
            "top_k": None,
        },
        "logging": {
            "log_level": "INFO",
        },
        "checkpointing": {
            "output_dir": "outputs/checkpoints",
            "save_every_n_epochs": 1,
            "checkpoint_name_prefix": "tiny_transformer",
        },
    }


# ---------------------------------------------------------------------------
# test_load_base_config
#
# This test checks that the real base config file loads correctly.
# ---------------------------------------------------------------------------
def test_load_base_config() -> None:
    config_path = _repo_root() / "configs" / "base.yaml"

    config = load_config(config_path)

    assert isinstance(config, AppConfig)
    assert config.project.name == "tiny-transformer"
    assert config.project.seed == 42
    assert config.data.context_window == 128
    assert config.model.embedding_dim == 128
    assert config.model.num_heads == 4
    assert config.training.batch_size == 32
    assert config.generation.max_new_tokens == 100
    assert config.logging.log_level == "INFO"


# ---------------------------------------------------------------------------
# test_load_debug_config
#
# This test checks that the real debug config file loads correctly.
# ---------------------------------------------------------------------------
def test_load_debug_config() -> None:
    config_path = _repo_root() / "configs" / "debug.yaml"

    config = load_config(config_path)

    assert isinstance(config, AppConfig)
    assert config.project.version == "0.1.0-debug"
    assert config.data.context_window == 32
    assert config.model.embedding_dim == 64
    assert config.model.num_layers == 1
    assert config.training.batch_size == 8
    assert config.generation.max_new_tokens == 20
    assert config.logging.log_level == "DEBUG"


# ---------------------------------------------------------------------------
# test_invalid_train_split_ratio_raises_error
#
# This test checks that an invalid train split ratio fails.
# ---------------------------------------------------------------------------
def test_invalid_train_split_ratio_raises_error(tmp_path: Path) -> None:
    config_dict = _valid_config_dict()
    config_dict["data"]["train_split_ratio"] = 1.2

    config_path = _write_yaml(tmp_path, "invalid_split.yaml", config_dict)

    with pytest.raises(ConfigError, match="train_split_ratio"):
        load_config(config_path)


# ---------------------------------------------------------------------------
# test_invalid_embedding_head_combination_raises_error
#
# This test checks that embedding size must divide evenly by head count.
# ---------------------------------------------------------------------------
def test_invalid_embedding_head_combination_raises_error(
    tmp_path: Path,
) -> None:
    config_dict = _valid_config_dict()
    config_dict["model"]["embedding_dim"] = 130
    config_dict["model"]["num_heads"] = 4

    config_path = _write_yaml(tmp_path, "invalid_heads.yaml", config_dict)

    with pytest.raises(ConfigError, match="embedding_dim"):
        load_config(config_path)


# ---------------------------------------------------------------------------
# test_invalid_dropout_rate_raises_error
#
# This test checks that dropout must stay in the valid range.
# ---------------------------------------------------------------------------
def test_invalid_dropout_rate_raises_error(tmp_path: Path) -> None:
    config_dict = _valid_config_dict()
    config_dict["model"]["dropout_rate"] = 1.5

    config_path = _write_yaml(tmp_path, "invalid_dropout.yaml", config_dict)

    with pytest.raises(ConfigError, match="dropout_rate"):
        load_config(config_path)


# ---------------------------------------------------------------------------
# test_invalid_temperature_raises_error
#
# This test checks that temperature must be greater than zero.
# ---------------------------------------------------------------------------
def test_invalid_temperature_raises_error(tmp_path: Path) -> None:
    config_dict = _valid_config_dict()
    config_dict["generation"]["temperature"] = 0.0

    config_path = _write_yaml(tmp_path, "invalid_temperature.yaml", config_dict)

    with pytest.raises(ConfigError, match="temperature"):
        load_config(config_path)


# ---------------------------------------------------------------------------
# test_invalid_top_k_raises_error
#
# This test checks that top_k must be positive when it is set.
# ---------------------------------------------------------------------------
def test_invalid_top_k_raises_error(tmp_path: Path) -> None:
    config_dict = _valid_config_dict()
    config_dict["generation"]["top_k"] = -5

    config_path = _write_yaml(tmp_path, "invalid_top_k.yaml", config_dict)

    with pytest.raises(ConfigError, match="top_k"):
        load_config(config_path)


# ---------------------------------------------------------------------------
# test_context_window_cannot_exceed_model_max_sequence_length
#
# This test checks a cross-section rule. The data window cannot be larger than
# the model sequence length.
# ---------------------------------------------------------------------------
def test_context_window_cannot_exceed_model_max_sequence_length(
    tmp_path: Path,
) -> None:
    config_dict = _valid_config_dict()
    config_dict["data"]["context_window"] = 256
    config_dict["model"]["max_sequence_length"] = 128

    config_path = _write_yaml(tmp_path, "invalid_context.yaml", config_dict)

    with pytest.raises(ConfigError, match="context_window"):
        load_config(config_path)


# ---------------------------------------------------------------------------
# test_missing_config_file_raises_error
#
# This test checks that loading a missing file fails clearly.
# ---------------------------------------------------------------------------
def test_missing_config_file_raises_error() -> None:
    missing_path = Path("configs/does_not_exist.yaml")

    with pytest.raises(ConfigError, match="does not exist"):
        load_config(missing_path)
