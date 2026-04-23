from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from tiny_transformer.config import AppConfig, load_config
from tiny_transformer.data.dataset import TokenSequenceDataset
from tiny_transformer.data.splits import split_token_ids
from tiny_transformer.data.tokenizer import CharacterTokenizer
from tiny_transformer.model.transformer import DecoderOnlyTransformer
from tiny_transformer.training.optimizer import build_optimizer
from tiny_transformer.training.trainer import Trainer


# ---------------------------------------------------------------------------
# _read_text
#
# This function reads the full training text from the configured dataset path.
# ---------------------------------------------------------------------------
def _read_text(dataset_path: str, encoding: str) -> str:
    path = Path(dataset_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset file was not found: {dataset_path}")

    if not path.is_file():
        raise FileNotFoundError(f"Dataset path is not a file: {dataset_path}")

    return path.read_text(encoding=encoding)


# ---------------------------------------------------------------------------
# _build_tokenizer
#
# This function builds a character tokenizer from dataset text.
# ---------------------------------------------------------------------------
def _build_tokenizer(text: str) -> CharacterTokenizer:
    return CharacterTokenizer.from_text(text)


# ---------------------------------------------------------------------------
# _resolve_vocab_size
#
# This function chooses the model vocab size.
# Config vocab size is used when it is provided.
# Otherwise, tokenizer vocab size is used.
# ---------------------------------------------------------------------------
def _resolve_vocab_size(
    config: AppConfig,
    tokenizer: CharacterTokenizer,
) -> int:
    if config.model.vocab_size is not None:
        return config.model.vocab_size

    return tokenizer.vocab_size


# ---------------------------------------------------------------------------
# _build_training_dataset
#
# This function creates the training dataset from raw text and config values.
# ---------------------------------------------------------------------------
def _build_training_dataset(
    config: AppConfig,
) -> tuple[TokenSequenceDataset, CharacterTokenizer]:
    text = _read_text(
        dataset_path=config.data.dataset_path,
        encoding=config.data.encoding,
    )
    tokenizer = _build_tokenizer(text)
    token_ids = tokenizer.encode(text)

    train_token_ids, _ = split_token_ids(
        token_ids=token_ids,
        train_split_ratio=config.data.train_split_ratio,
    )

    dataset = TokenSequenceDataset(
        token_ids=train_token_ids,
        context_window=config.data.context_window,
    )

    return dataset, tokenizer


# ---------------------------------------------------------------------------
# _collate_token_batches
#
# This function turns a batch of list-based dataset samples into 2D long
# tensors.
# ---------------------------------------------------------------------------
def _collate_token_batches(
    batch: list[tuple[list[int], list[int]]],
) -> tuple[torch.Tensor, torch.Tensor]:
    input_id_rows = [sample[0] for sample in batch]
    target_id_rows = [sample[1] for sample in batch]

    input_ids = torch.tensor(input_id_rows, dtype=torch.long)
    target_ids = torch.tensor(target_id_rows, dtype=torch.long)

    return input_ids, target_ids


# ---------------------------------------------------------------------------
# _build_training_dataloader
#
# This function wraps the training dataset in a dataloader.
# ---------------------------------------------------------------------------
def _build_training_dataloader(
    config: AppConfig,
    dataset: TokenSequenceDataset,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=_collate_token_batches,
    )


# ---------------------------------------------------------------------------
# _build_model
#
# This function builds the decoder-only transformer from config values.
# ---------------------------------------------------------------------------
def _build_model(
    config: AppConfig,
    vocab_size: int,
) -> DecoderOnlyTransformer:
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
# run_training
#
# This function wires the training components together and runs the configured
# number of training epochs.
# ---------------------------------------------------------------------------
def run_training(config_path: str = "configs/base.yaml") -> None:
    config = load_config(config_path)
    dataset, tokenizer = _build_training_dataset(config)
    dataloader = _build_training_dataloader(config, dataset)
    vocab_size = _resolve_vocab_size(config, tokenizer)
    model = _build_model(config, vocab_size)
    optimizer = build_optimizer(model, config)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=config.training.device,
    )

    for epoch_index in range(config.training.max_epochs):
        epoch_result = trainer.train_epoch(dataloader)
        epoch_number = epoch_index + 1

        print(
            f"Epoch {epoch_number}/{config.training.max_epochs} "
            f"- average_loss: {epoch_result.average_loss:.4f} "
            f"- batch_count: {epoch_result.batch_count}"
        )


# ---------------------------------------------------------------------------
# _parse_args
#
# This function reads the config path from the command line.
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Tiny Transformer training.",
    )
    parser.add_argument(
        "--config",
        default="configs/base.yaml",
        help="Path to the YAML config file.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# main
#
# This function runs the training script.
# ---------------------------------------------------------------------------
def main() -> None:
    args = _parse_args()
    run_training(config_path=args.config)


if __name__ == "__main__":
    main()
