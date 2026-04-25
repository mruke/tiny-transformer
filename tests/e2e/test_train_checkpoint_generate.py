from __future__ import annotations

from pathlib import Path

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
from tiny_transformer.data.dataset import TokenSequenceDataset
from tiny_transformer.data.tokenizer import CharacterTokenizer
from tiny_transformer.inference.generator import generate_next_tokens_greedy
from tiny_transformer.model.transformer import DecoderOnlyTransformer
from tiny_transformer.training.checkpoints import build_checkpoint_path
from tiny_transformer.training.optimizer import build_optimizer
from tiny_transformer.training.trainer import Trainer


# ---------------------------------------------------------------------------
# _build_e2e_config
#
# This helper builds a small config for the end-to-end smoke test.
# It keeps the model tiny so the full workflow can run quickly.
# ---------------------------------------------------------------------------
def _build_e2e_config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        project=ProjectConfig(
            name="tiny-transformer-e2e",
            version="0.5.0-test",
            seed=123,
        ),
        data=DataConfig(
            dataset_path="unused-e2e-corpus.txt",
            train_split_ratio=0.8,
            context_window=4,
            encoding="utf-8",
        ),
        model=ModelConfig(
            vocab_size=None,
            max_sequence_length=4,
            embedding_dim=8,
            num_heads=2,
            num_layers=1,
            feedforward_dim=16,
            dropout_rate=0.0,
        ),
        training=TrainingConfig(
            batch_size=1,
            learning_rate=0.001,
            weight_decay=0.0,
            max_epochs=1,
            device="cpu",
            eval_interval=1,
            log_interval=1,
        ),
        generation=GenerationConfig(
            max_new_tokens=2,
            temperature=1.0,
            top_k=None,
        ),
        logging=LoggingConfig(log_level="INFO"),
        checkpointing=CheckpointingConfig(
            output_dir=str(tmp_path / "checkpoints"),
            save_every_n_epochs=1,
            checkpoint_name_prefix="tiny_transformer_e2e",
        ),
    )


# ---------------------------------------------------------------------------
# _build_model
#
# This helper builds a tiny decoder-only transformer for the E2E workflow.
# It uses the same config values for training, checkpointing, and generation.
# ---------------------------------------------------------------------------
def _build_model(config: AppConfig, vocab_size: int) -> DecoderOnlyTransformer:
    return DecoderOnlyTransformer(
        vocab_size=vocab_size,
        max_sequence_length=config.model.max_sequence_length,
        embedding_dim=config.model.embedding_dim,
        num_heads=config.model.num_heads,
        num_layers=config.model.num_layers,
        feedforward_dim=config.model.feedforward_dim,
        dropout_rate=config.model.dropout_rate,
    )


# ---------------------------------------------------------------------------
# test_train_checkpoint_and_generate_end_to_end
#
# This test runs a small full workflow:
# train one batch, save a checkpoint, load it into a fresh model, and generate.
# ---------------------------------------------------------------------------
def test_train_checkpoint_and_generate_end_to_end(tmp_path: Path) -> None:
    config = _build_e2e_config(tmp_path)
    torch.manual_seed(config.project.seed)

    corpus = "abcdabcdabcd"
    tokenizer = CharacterTokenizer.from_text(corpus)
    token_ids = tokenizer.encode(corpus)

    dataset = TokenSequenceDataset(
        token_ids=token_ids,
        context_window=config.data.context_window,
    )

    input_ids, target_ids = dataset[0]
    input_batch = torch.tensor([input_ids], dtype=torch.long)
    target_batch = torch.tensor([target_ids], dtype=torch.long)

    model = _build_model(
        config=config,
        vocab_size=tokenizer.vocab_size,
    )
    optimizer = build_optimizer(
        model=model,
        config=config,
    )
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=config.training.device,
    )

    train_result = trainer.train_epoch([(input_batch, target_batch)])

    assert train_result.batch_count == 1
    assert train_result.average_loss > 0.0

    checkpoint_path = build_checkpoint_path(
        output_dir=config.checkpointing.output_dir,
        checkpoint_prefix=config.checkpointing.checkpoint_name_prefix,
        epoch=1,
    )
    saved_checkpoint_path = trainer.save_checkpoint(
        config=config,
        epoch=1,
        checkpoint_path=checkpoint_path,
    )

    assert saved_checkpoint_path.exists()

    reloaded_model = _build_model(
        config=config,
        vocab_size=tokenizer.vocab_size,
    )
    reloaded_optimizer = build_optimizer(
        model=reloaded_model,
        config=config,
    )
    reloaded_trainer = Trainer(
        model=reloaded_model,
        optimizer=reloaded_optimizer,
        device=config.training.device,
    )

    metadata = reloaded_trainer.load_checkpoint(saved_checkpoint_path)

    assert metadata.epoch == 1
    assert metadata.project_name == config.project.name
    assert metadata.project_version == config.project.version

    prompt_token_ids = torch.tensor(
        [tokenizer.encode("ab")],
        dtype=torch.long,
    )
    generated_token_ids = generate_next_tokens_greedy(
        model=reloaded_model,
        prompt_token_ids=prompt_token_ids,
        max_new_tokens=config.generation.max_new_tokens,
    )

    generated_ids = generated_token_ids[0].tolist()
    generated_text = tokenizer.decode(generated_ids)

    assert generated_token_ids.shape == (1, 4)
    assert generated_text.startswith("ab")
    assert len(generated_text) == 4
