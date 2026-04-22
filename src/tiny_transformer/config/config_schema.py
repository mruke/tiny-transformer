from __future__ import annotations

from dataclasses import dataclass

from tiny_transformer.config.config_exceptions import ConfigError
from tiny_transformer.config.config_validation import (
    _require_non_empty_string,
    _require_non_negative_int,
    _require_non_negative_number,
    _require_positive_int,
    _require_positive_number,
)


# ---------------------------------------------------------------------------
# ProjectConfig
#
# ProjectConfig stores basic project-wide values.
# These values describe the project and help keep runs repeatable.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ProjectConfig:
    name: str
    version: str
    seed: int

    # -----------------------------------------------------------------------
    # ProjectConfig.__post_init__
    #
    # This method validates the project config right after it is created.
    # -----------------------------------------------------------------------
    def __post_init__(self) -> None:
        _require_non_empty_string(self.name, "project.name")
        _require_non_empty_string(self.version, "project.version")
        _require_non_negative_int(self.seed, "project.seed")


# ---------------------------------------------------------------------------
# DataConfig
#
# DataConfig stores settings for reading text and building training samples.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DataConfig:
    dataset_path: str
    train_split_ratio: float
    context_window: int
    encoding: str

    # -----------------------------------------------------------------------
    # DataConfig.__post_init__
    #
    # This method validates the data config right after it is created.
    # -----------------------------------------------------------------------
    def __post_init__(self) -> None:
        _require_non_empty_string(self.dataset_path, "data.dataset_path")

        if not (0.0 < self.train_split_ratio < 1.0):
            raise ConfigError(
                "'data.train_split_ratio' must be greater than 0 and less than 1."
            )

        _require_positive_int(self.context_window, "data.context_window")
        _require_non_empty_string(self.encoding, "data.encoding")


# ---------------------------------------------------------------------------
# ModelConfig
#
# ModelConfig stores the transformer shape.
# These values describe how large the model is and how it is structured.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int | None
    max_sequence_length: int
    embedding_dim: int
    num_heads: int
    num_layers: int
    feedforward_dim: int
    dropout_rate: float

    # -----------------------------------------------------------------------
    # ModelConfig.__post_init__
    #
    # This method validates the model config right after it is created.
    # -----------------------------------------------------------------------
    def __post_init__(self) -> None:
        if self.vocab_size is not None:
            _require_positive_int(self.vocab_size, "model.vocab_size")

        _require_positive_int(
            self.max_sequence_length,
            "model.max_sequence_length",
        )
        _require_positive_int(self.embedding_dim, "model.embedding_dim")
        _require_positive_int(self.num_heads, "model.num_heads")
        _require_positive_int(self.num_layers, "model.num_layers")
        _require_positive_int(
            self.feedforward_dim,
            "model.feedforward_dim",
        )

        if not isinstance(self.dropout_rate, (int, float)) or not (
            0.0 <= self.dropout_rate < 1.0
        ):
            raise ConfigError("'model.dropout_rate' must be between 0.0 and 1.0.")

        # Each attention head gets an equal share of the embedding size.
        if self.embedding_dim % self.num_heads != 0:
            raise ConfigError(
                "'model.embedding_dim' must be divisible by 'model.num_heads'."
            )


# ---------------------------------------------------------------------------
# TrainingConfig
#
# TrainingConfig stores settings for optimization and training flow.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int
    learning_rate: float
    weight_decay: float
    max_epochs: int
    device: str
    eval_interval: int
    log_interval: int

    # -----------------------------------------------------------------------
    # TrainingConfig.__post_init__
    #
    # This method validates the training config right after it is created.
    # -----------------------------------------------------------------------
    def __post_init__(self) -> None:
        _require_positive_int(self.batch_size, "training.batch_size")
        _require_positive_number(self.learning_rate, "training.learning_rate")
        _require_non_negative_number(
            self.weight_decay,
            "training.weight_decay",
        )
        _require_positive_int(self.max_epochs, "training.max_epochs")
        _require_non_empty_string(self.device, "training.device")
        _require_positive_int(self.eval_interval, "training.eval_interval")
        _require_positive_int(self.log_interval, "training.log_interval")


# ---------------------------------------------------------------------------
# GenerationConfig
#
# GenerationConfig stores settings for text generation.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int
    temperature: float
    top_k: int | None

    # -----------------------------------------------------------------------
    # GenerationConfig.__post_init__
    #
    # This method validates the generation config right after it is created.
    # -----------------------------------------------------------------------
    def __post_init__(self) -> None:
        _require_positive_int(
            self.max_new_tokens,
            "generation.max_new_tokens",
        )
        _require_positive_number(
            self.temperature,
            "generation.temperature",
        )

        if self.top_k is not None:
            _require_positive_int(self.top_k, "generation.top_k")


# ---------------------------------------------------------------------------
# LoggingConfig
#
# LoggingConfig stores simple logging settings.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LoggingConfig:
    log_level: str

    # -----------------------------------------------------------------------
    # LoggingConfig.__post_init__
    #
    # This method validates the logging config right after it is created.
    # -----------------------------------------------------------------------
    def __post_init__(self) -> None:
        _require_non_empty_string(self.log_level, "logging.log_level")


# ---------------------------------------------------------------------------
# CheckpointingConfig
#
# CheckpointingConfig stores settings for saving model state to disk.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CheckpointingConfig:
    output_dir: str
    save_every_n_epochs: int
    checkpoint_name_prefix: str

    # -----------------------------------------------------------------------
    # CheckpointingConfig.__post_init__
    #
    # This method validates the checkpointing config right after it is created.
    # -----------------------------------------------------------------------
    def __post_init__(self) -> None:
        _require_non_empty_string(
            self.output_dir,
            "checkpointing.output_dir",
        )
        _require_positive_int(
            self.save_every_n_epochs,
            "checkpointing.save_every_n_epochs",
        )
        _require_non_empty_string(
            self.checkpoint_name_prefix,
            "checkpointing.checkpoint_name_prefix",
        )


# ---------------------------------------------------------------------------
# AppConfig
#
# AppConfig is the top-level config object.
# It groups all smaller config sections into one object.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AppConfig:
    project: ProjectConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    generation: GenerationConfig
    logging: LoggingConfig
    checkpointing: CheckpointingConfig

    # -----------------------------------------------------------------------
    # AppConfig.__post_init__
    #
    # This method checks rules that involve more than one section.
    # -----------------------------------------------------------------------
    def __post_init__(self) -> None:
        # The model must support at least as much sequence length
        # as the dataset pipeline is going to use.
        if self.data.context_window > self.model.max_sequence_length:
            raise ConfigError(
                "'data.context_window' cannot exceed 'model.max_sequence_length'."
            )
