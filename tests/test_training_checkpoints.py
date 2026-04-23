from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tiny_transformer.config import (
    AppConfig,
    CheckpointingConfig,
    DataConfig,
    GenerationConfig,
    LoggingConfig,
    ModelConfig,
    ProjectConfig,
    TrainingConfig,
)
from tiny_transformer.training.checkpoints import (
    CheckpointMetadata,
    build_checkpoint_path,
    load_checkpoint,
    save_checkpoint,
)
from tiny_transformer.training.optimizer import build_optimizer
from tiny_transformer.training.trainer import Trainer


# ---------------------------------------------------------------------------
# _build_test_config
#
# This helper creates a valid app config for checkpoint tests.
# ---------------------------------------------------------------------------
def _build_test_config() -> AppConfig:
    return AppConfig(
        project=ProjectConfig(
            name="tiny-transformer",
            version="0.1.0-test",
            seed=42,
        ),
        data=DataConfig(
            dataset_path="data/input.txt",
            train_split_ratio=0.9,
            context_window=8,
            encoding="utf-8",
        ),
        model=ModelConfig(
            vocab_size=16,
            max_sequence_length=8,
            embedding_dim=8,
            num_heads=2,
            num_layers=1,
            feedforward_dim=32,
            dropout_rate=0.1,
        ),
        training=TrainingConfig(
            batch_size=2,
            learning_rate=0.001,
            weight_decay=0.0,
            max_epochs=2,
            device="cpu",
            eval_interval=10,
            log_interval=10,
        ),
        generation=GenerationConfig(
            max_new_tokens=10,
            temperature=1.0,
            top_k=None,
        ),
        logging=LoggingConfig(
            log_level="INFO",
        ),
        checkpointing=CheckpointingConfig(
            output_dir="outputs/checkpoints-test",
            save_every_n_epochs=1,
            checkpoint_name_prefix="tiny_transformer_test",
        ),
    )


# ---------------------------------------------------------------------------
# _build_test_model
#
# This helper creates a small model for checkpoint tests.
# ---------------------------------------------------------------------------
def _build_test_model() -> torch.nn.Module:
    return torch.nn.Linear(8, 4)


# ---------------------------------------------------------------------------
# _build_test_trainer
#
# This helper creates a trainer for checkpoint integration tests.
# ---------------------------------------------------------------------------
def _build_test_trainer(config: AppConfig) -> Trainer:
    model = _build_test_model()
    optimizer = build_optimizer(model, config)

    return Trainer(
        model=model,
        optimizer=optimizer,
        device=config.training.device,
    )


# ---------------------------------------------------------------------------
# test_build_checkpoint_path_returns_expected_path
#
# This test checks that checkpoint path naming is consistent.
# ---------------------------------------------------------------------------
def test_build_checkpoint_path_returns_expected_path(tmp_path: Path) -> None:
    checkpoint_path = build_checkpoint_path(
        output_dir=tmp_path,
        checkpoint_prefix="tiny_transformer",
        epoch=3,
    )

    assert checkpoint_path == tmp_path / "tiny_transformer_epoch_3.pt"


# ---------------------------------------------------------------------------
# test_save_checkpoint_creates_checkpoint_file
#
# This test checks that saving a checkpoint creates the target file.
# ---------------------------------------------------------------------------
def test_save_checkpoint_creates_checkpoint_file(tmp_path: Path) -> None:
    config = _build_test_config()
    model = _build_test_model()
    optimizer = build_optimizer(model, config)
    checkpoint_path = tmp_path / "checkpoint.pt"

    saved_path = save_checkpoint(
        model=model,
        optimizer=optimizer,
        config=config,
        epoch=1,
        checkpoint_path=checkpoint_path,
    )

    assert saved_path == checkpoint_path
    assert checkpoint_path.exists()


# ---------------------------------------------------------------------------
# test_load_checkpoint_restores_model_parameters
#
# This test checks that loading a checkpoint restores model parameters.
# ---------------------------------------------------------------------------
def test_load_checkpoint_restores_model_parameters(tmp_path: Path) -> None:
    config = _build_test_config()
    model = _build_test_model()
    optimizer = build_optimizer(model, config)
    checkpoint_path = tmp_path / "checkpoint.pt"

    original_parameters = [
        parameter.detach().clone() for parameter in model.parameters()
    ]

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        config=config,
        epoch=1,
        checkpoint_path=checkpoint_path,
    )

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)

    load_checkpoint(
        model=model,
        optimizer=optimizer,
        checkpoint_path=checkpoint_path,
    )

    restored_parameters = list(model.parameters())

    assert all(
        torch.equal(original_parameter, restored_parameter)
        for original_parameter, restored_parameter in zip(
            original_parameters,
            restored_parameters,
            strict=True,
        )
    )


# ---------------------------------------------------------------------------
# test_load_checkpoint_returns_saved_metadata
#
# This test checks that loading a checkpoint returns the saved metadata.
# ---------------------------------------------------------------------------
def test_load_checkpoint_returns_saved_metadata(tmp_path: Path) -> None:
    config = _build_test_config()
    model = _build_test_model()
    optimizer = build_optimizer(model, config)
    checkpoint_path = tmp_path / "checkpoint.pt"

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        config=config,
        epoch=2,
        checkpoint_path=checkpoint_path,
    )

    metadata = load_checkpoint(
        model=model,
        optimizer=optimizer,
        checkpoint_path=checkpoint_path,
    )

    assert isinstance(metadata, CheckpointMetadata)
    assert metadata.epoch == 2
    assert metadata.project_name == "tiny-transformer"
    assert metadata.project_version == "0.1.0-test"


# ---------------------------------------------------------------------------
# test_load_checkpoint_rejects_missing_file
#
# This test checks that loading a missing checkpoint fails clearly.
# ---------------------------------------------------------------------------
def test_load_checkpoint_rejects_missing_file(tmp_path: Path) -> None:
    config = _build_test_config()
    model = _build_test_model()
    optimizer = build_optimizer(model, config)
    checkpoint_path = tmp_path / "missing.pt"

    with pytest.raises(FileNotFoundError, match="Checkpoint file was not found"):
        load_checkpoint(
            model=model,
            optimizer=optimizer,
            checkpoint_path=checkpoint_path,
        )


# ---------------------------------------------------------------------------
# test_save_checkpoint_rejects_non_positive_epoch
#
# This test checks that epoch must be a positive integer.
# ---------------------------------------------------------------------------
def test_save_checkpoint_rejects_non_positive_epoch(tmp_path: Path) -> None:
    config = _build_test_config()
    model = _build_test_model()
    optimizer = build_optimizer(model, config)
    checkpoint_path = tmp_path / "checkpoint.pt"

    with pytest.raises(ValueError, match="epoch must be a positive integer"):
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            config=config,
            epoch=0,
            checkpoint_path=checkpoint_path,
        )


# ---------------------------------------------------------------------------
# test_trainer_save_checkpoint_creates_checkpoint_file
#
# This test checks that the trainer can save its current state through the
# checkpoint module.
# ---------------------------------------------------------------------------
def test_trainer_save_checkpoint_creates_checkpoint_file(tmp_path: Path) -> None:
    config = _build_test_config()
    trainer = _build_test_trainer(config)
    checkpoint_path = tmp_path / "trainer_checkpoint.pt"

    saved_path = trainer.save_checkpoint(
        config=config,
        epoch=1,
        checkpoint_path=checkpoint_path,
    )

    assert saved_path == checkpoint_path
    assert checkpoint_path.exists()


# ---------------------------------------------------------------------------
# test_trainer_load_checkpoint_restores_saved_metadata
#
# This test checks that the trainer can load checkpoint state and receive the
# saved metadata.
# ---------------------------------------------------------------------------
def test_trainer_load_checkpoint_restores_saved_metadata(tmp_path: Path) -> None:
    config = _build_test_config()
    trainer = _build_test_trainer(config)
    checkpoint_path = tmp_path / "trainer_checkpoint.pt"

    trainer.save_checkpoint(
        config=config,
        epoch=2,
        checkpoint_path=checkpoint_path,
    )

    loaded_metadata = trainer.load_checkpoint(
        checkpoint_path=checkpoint_path,
    )

    assert isinstance(loaded_metadata, CheckpointMetadata)
    assert loaded_metadata.epoch == 2
    assert loaded_metadata.project_name == "tiny-transformer"
    assert loaded_metadata.project_version == "0.1.0-test"
