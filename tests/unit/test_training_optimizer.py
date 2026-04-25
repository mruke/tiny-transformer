from __future__ import annotations

import pytest
from torch import nn
from torch.optim import AdamW

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
from tiny_transformer.training.optimizer import build_optimizer


# ---------------------------------------------------------------------------
# _build_test_config
#
# This helper creates a valid app config for optimizer tests.
# ---------------------------------------------------------------------------
def _build_test_config(
    learning_rate: float = 0.001,
    weight_decay: float = 0.0,
) -> AppConfig:
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
            vocab_size=32,
            max_sequence_length=8,
            embedding_dim=16,
            num_heads=4,
            num_layers=2,
            feedforward_dim=64,
            dropout_rate=0.1,
        ),
        training=TrainingConfig(
            batch_size=4,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
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
# test_build_optimizer_returns_adamw_instance
#
# This test checks that optimizer creation returns an AdamW optimizer.
# ---------------------------------------------------------------------------
def test_build_optimizer_returns_adamw_instance() -> None:
    model = nn.Linear(8, 4)
    config = _build_test_config()

    optimizer = build_optimizer(model, config)

    assert isinstance(optimizer, AdamW)


# ---------------------------------------------------------------------------
# test_build_optimizer_uses_training_learning_rate
#
# This test checks that the optimizer uses the configured learning rate.
# ---------------------------------------------------------------------------
def test_build_optimizer_uses_training_learning_rate() -> None:
    model = nn.Linear(8, 4)
    config = _build_test_config(learning_rate=0.0005)

    optimizer = build_optimizer(model, config)

    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0005)


# ---------------------------------------------------------------------------
# test_build_optimizer_uses_training_weight_decay
#
# This test checks that the optimizer uses the configured weight decay.
# ---------------------------------------------------------------------------
def test_build_optimizer_uses_training_weight_decay() -> None:
    model = nn.Linear(8, 4)
    config = _build_test_config(weight_decay=0.02)

    optimizer = build_optimizer(model, config)

    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.02)


# ---------------------------------------------------------------------------
# test_build_optimizer_uses_only_trainable_parameters
#
# This test checks that frozen parameters are skipped.
# ---------------------------------------------------------------------------
def test_build_optimizer_uses_only_trainable_parameters() -> None:
    model = nn.Sequential(
        nn.Linear(8, 4),
        nn.Linear(4, 2),
    )

    for parameter in model[0].parameters():
        parameter.requires_grad = False

    config = _build_test_config()

    optimizer = build_optimizer(model, config)

    optimized_parameters = optimizer.param_groups[0]["params"]
    expected_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]

    assert optimized_parameters == expected_parameters


# ---------------------------------------------------------------------------
# test_build_optimizer_rejects_non_module_model
#
# This test checks that the model must be a PyTorch module.
# ---------------------------------------------------------------------------
def test_build_optimizer_rejects_non_module_model() -> None:
    config = _build_test_config()

    with pytest.raises(
        TypeError,
        match="model must be an instance of torch.nn.Module",
    ):
        build_optimizer(object(), config)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# test_build_optimizer_rejects_model_without_trainable_parameters
#
# This test checks that the optimizer cannot be built when all parameters
# are frozen.
# ---------------------------------------------------------------------------
def test_build_optimizer_rejects_model_without_trainable_parameters() -> None:
    model = nn.Linear(8, 4)

    for parameter in model.parameters():
        parameter.requires_grad = False

    config = _build_test_config()

    with pytest.raises(
        ValueError,
        match="model must have at least one parameter to optimize",
    ):
        build_optimizer(model, config)
