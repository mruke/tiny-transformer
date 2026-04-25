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
# This helper creates a valid app config for trainer tests.
# ---------------------------------------------------------------------------
def _build_test_config() -> AppConfig:
    return AppConfig(
        project=ProjectConfig(
            name="tiny-transformer",
            version="0.5.0-test",
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
# This helper creates a small transformer for trainer tests.
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
# _build_training_batch
#
# This helper creates one small training batch.
# ---------------------------------------------------------------------------
def _build_training_batch() -> tuple[torch.Tensor, torch.Tensor]:
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
# _build_training_batches
#
# This helper creates a small list of training batches for one epoch test.
# ---------------------------------------------------------------------------
def _build_training_batches() -> list[tuple[torch.Tensor, torch.Tensor]]:
    first_batch = _build_training_batch()

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
# test_trainer_train_batch_returns_finite_loss
#
# This test checks that one training batch returns a finite loss value.
# ---------------------------------------------------------------------------
def test_trainer_train_batch_returns_finite_loss() -> None:
    config = _build_test_config()
    model = _build_test_model()
    optimizer = build_optimizer(model, config)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=config.training.device,
    )
    input_ids, target_ids = _build_training_batch()

    loss_value = trainer.train_batch(input_ids, target_ids)

    assert isinstance(loss_value, float)
    assert torch.isfinite(torch.tensor(loss_value))


# ---------------------------------------------------------------------------
# test_trainer_train_batch_updates_model_parameters
#
# This test checks that one training batch updates at least one model
# parameter.
# ---------------------------------------------------------------------------
def test_trainer_train_batch_updates_model_parameters() -> None:
    config = _build_test_config()
    model = _build_test_model()
    optimizer = build_optimizer(model, config)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=config.training.device,
    )
    input_ids, target_ids = _build_training_batch()

    parameters_before_step = [
        parameter.detach().clone() for parameter in model.parameters()
    ]

    trainer.train_batch(input_ids, target_ids)

    parameters_after_step = list(model.parameters())

    assert any(
        not torch.equal(before_step, after_step)
        for before_step, after_step in zip(
            parameters_before_step,
            parameters_after_step,
            strict=True,
        )
    )


# ---------------------------------------------------------------------------
# test_trainer_train_epoch_returns_epoch_result
#
# This test checks that one training epoch returns average loss and batch
# count.
# ---------------------------------------------------------------------------
def test_trainer_train_epoch_returns_epoch_result() -> None:
    config = _build_test_config()
    model = _build_test_model()
    optimizer = build_optimizer(model, config)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=config.training.device,
    )
    batches = _build_training_batches()

    epoch_result = trainer.train_epoch(batches)

    assert isinstance(epoch_result, TrainEpochResult)
    assert epoch_result.batch_count == 2
    assert isinstance(epoch_result.average_loss, float)
    assert torch.isfinite(torch.tensor(epoch_result.average_loss))


# ---------------------------------------------------------------------------
# test_trainer_train_epoch_rejects_empty_batches
#
# This test checks that a training epoch must process at least one batch.
# ---------------------------------------------------------------------------
def test_trainer_train_epoch_rejects_empty_batches() -> None:
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
        trainer.train_epoch([])


# ---------------------------------------------------------------------------
# test_trainer_rejects_non_module_model
#
# This test checks that the trainer requires a PyTorch module.
# ---------------------------------------------------------------------------
def test_trainer_rejects_non_module_model() -> None:
    optimizer = build_optimizer(_build_test_model(), _build_test_config())

    with pytest.raises(
        TypeError,
        match="model must be an instance of torch.nn.Module",
    ):
        Trainer(
            model=object(),  # type: ignore[arg-type]
            optimizer=optimizer,
            device="cpu",
        )


# ---------------------------------------------------------------------------
# test_trainer_rejects_non_optimizer
#
# This test checks that the trainer requires a PyTorch optimizer.
# ---------------------------------------------------------------------------
def test_trainer_rejects_non_optimizer() -> None:
    with pytest.raises(
        TypeError,
        match="optimizer must be an instance of torch.optim.Optimizer",
    ):
        Trainer(
            model=_build_test_model(),
            optimizer=object(),  # type: ignore[arg-type]
            device="cpu",
        )


# ---------------------------------------------------------------------------
# test_trainer_rejects_mismatched_input_and_target_shape
#
# This test checks that input IDs and target IDs must have the same shape.
# ---------------------------------------------------------------------------
def test_trainer_rejects_mismatched_input_and_target_shape() -> None:
    config = _build_test_config()
    model = _build_test_model()
    optimizer = build_optimizer(model, config)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=config.training.device,
    )
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    target_ids = torch.tensor([[2, 3], [5, 6]], dtype=torch.long)

    with pytest.raises(
        ValueError,
        match="input_ids and target_ids must have the same shape",
    ):
        trainer.train_batch(input_ids, target_ids)


# ---------------------------------------------------------------------------
# test_trainer_rejects_non_long_input_ids
#
# This test checks that input IDs must use torch.long dtype.
# ---------------------------------------------------------------------------
def test_trainer_rejects_non_long_input_ids() -> None:
    config = _build_test_config()
    model = _build_test_model()
    optimizer = build_optimizer(model, config)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=config.training.device,
    )
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32)
    target_ids = torch.tensor([[2, 3, 4], [5, 6, 7]], dtype=torch.long)

    with pytest.raises(TypeError, match="input_ids must use torch.long dtype"):
        trainer.train_batch(input_ids, target_ids)
