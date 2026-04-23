from __future__ import annotations

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
from tiny_transformer.model.transformer import DecoderOnlyTransformer
from tiny_transformer.training.optimizer import build_optimizer
from tiny_transformer.training.trainer import TrainEpochResult, Trainer


# ---------------------------------------------------------------------------
# _build_test_config
#
# This helper creates a valid app config for validation tests.
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
            num_layers=2,
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
            output_dir="outputs/checkpoints",
            save_every_n_epochs=1,
            checkpoint_name_prefix="tiny_transformer",
        ),
    )


# ---------------------------------------------------------------------------
# _build_test_model
#
# This helper creates a small transformer for validation tests.
# ---------------------------------------------------------------------------
def _build_test_model() -> DecoderOnlyTransformer:
    return DecoderOnlyTransformer(
        vocab_size=16,
        max_sequence_length=8,
        embedding_dim=8,
        num_heads=2,
        num_layers=2,
        feedforward_dim=32,
        dropout_rate=0.1,
    )


# ---------------------------------------------------------------------------
# _build_validation_batch
#
# This helper creates one small validation batch.
# ---------------------------------------------------------------------------
def _build_validation_batch() -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = torch.tensor(
        [
            [1, 2, 3, 4],
            [2, 3, 4, 5],
        ],
        dtype=torch.long,
    )
    target_ids = torch.tensor(
        [
            [2, 3, 4, 5],
            [3, 4, 5, 6],
        ],
        dtype=torch.long,
    )

    return input_ids, target_ids


# ---------------------------------------------------------------------------
# _build_validation_batches
#
# This helper creates a small list of validation batches for one epoch test.
# ---------------------------------------------------------------------------
def _build_validation_batches() -> list[tuple[torch.Tensor, torch.Tensor]]:
    first_batch = _build_validation_batch()

    second_batch = (
        torch.tensor(
            [
                [3, 4, 5, 6],
                [4, 5, 6, 7],
            ],
            dtype=torch.long,
        ),
        torch.tensor(
            [
                [4, 5, 6, 7],
                [5, 6, 7, 8],
            ],
            dtype=torch.long,
        ),
    )

    return [first_batch, second_batch]


# ---------------------------------------------------------------------------
# test_trainer_validate_batch_returns_finite_loss
#
# This test checks that one validation batch returns a finite loss value.
# ---------------------------------------------------------------------------
def test_trainer_validate_batch_returns_finite_loss() -> None:
    config = _build_test_config()
    model = _build_test_model()
    optimizer = build_optimizer(model, config)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=config.training.device,
    )
    input_ids, target_ids = _build_validation_batch()

    loss_value = trainer.validate_batch(input_ids, target_ids)

    assert isinstance(loss_value, float)
    assert torch.isfinite(torch.tensor(loss_value))


# ---------------------------------------------------------------------------
# test_trainer_validate_batch_does_not_update_model_parameters
#
# This test checks that validation does not change model parameters.
# ---------------------------------------------------------------------------
def test_trainer_validate_batch_does_not_update_model_parameters() -> None:
    config = _build_test_config()
    model = _build_test_model()
    optimizer = build_optimizer(model, config)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=config.training.device,
    )
    input_ids, target_ids = _build_validation_batch()

    parameters_before_validation = [
        parameter.detach().clone() for parameter in model.parameters()
    ]

    trainer.validate_batch(input_ids, target_ids)

    parameters_after_validation = list(model.parameters())

    assert all(
        torch.equal(before_validation, after_validation)
        for before_validation, after_validation in zip(
            parameters_before_validation,
            parameters_after_validation,
            strict=True,
        )
    )


# ---------------------------------------------------------------------------
# test_trainer_validate_batch_does_not_create_gradients
#
# This test checks that validation does not backpropagate gradients.
# ---------------------------------------------------------------------------
def test_trainer_validate_batch_does_not_create_gradients() -> None:
    config = _build_test_config()
    model = _build_test_model()
    optimizer = build_optimizer(model, config)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=config.training.device,
    )
    input_ids, target_ids = _build_validation_batch()

    trainer.validate_batch(input_ids, target_ids)

    assert all(parameter.grad is None for parameter in model.parameters())


# ---------------------------------------------------------------------------
# test_trainer_validate_epoch_returns_epoch_result
#
# This test checks that one validation epoch returns average loss and batch
# count.
# ---------------------------------------------------------------------------
def test_trainer_validate_epoch_returns_epoch_result() -> None:
    config = _build_test_config()
    model = _build_test_model()
    optimizer = build_optimizer(model, config)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=config.training.device,
    )
    batches = _build_validation_batches()

    epoch_result = trainer.validate_epoch(batches)

    assert isinstance(epoch_result, TrainEpochResult)
    assert epoch_result.batch_count == 2
    assert isinstance(epoch_result.average_loss, float)
    assert torch.isfinite(torch.tensor(epoch_result.average_loss))


# ---------------------------------------------------------------------------
# test_trainer_validate_epoch_rejects_empty_batches
#
# This test checks that a validation epoch must process at least one batch.
# ---------------------------------------------------------------------------
def test_trainer_validate_epoch_rejects_empty_batches() -> None:
    config = _build_test_config()
    model = _build_test_model()
    optimizer = build_optimizer(model, config)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=config.training.device,
    )

    with pytest.raises(
        ValueError,
        match="Training epoch must process at least one batch",
    ):
        trainer.validate_epoch([])
