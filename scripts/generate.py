from __future__ import annotations

import argparse
from pathlib import Path

import torch

from tiny_transformer.config import AppConfig, load_config
from tiny_transformer.data.tokenizer import CharacterTokenizer
from tiny_transformer.model.transformer import DecoderOnlyTransformer
from tiny_transformer.training.checkpoints import build_checkpoint_path
from tiny_transformer.training.optimizer import build_optimizer
from tiny_transformer.training.trainer import Trainer
from tiny_transformer.inference.generator import generate_next_tokens_greedy


# ---------------------------------------------------------------------------
# _parse_args
#
# This function reads generation script arguments from the command line.
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Tiny Transformer text generation.",
    )
    parser.add_argument(
        "--config",
        default="configs/base.yaml",
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="Prompt text used to begin generation.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional path to a checkpoint file. If omitted, the script uses the checkpoint for the final configured epoch.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# _read_text
#
# This function reads the full corpus text from the configured dataset path.
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
# This function builds a character tokenizer from corpus text.
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
# _resolve_checkpoint_path
#
# This function returns the checkpoint path used for generation.
# If the caller does not provide a path, the final configured epoch checkpoint
# is used.
# ---------------------------------------------------------------------------
def _resolve_checkpoint_path(
    config: AppConfig,
    checkpoint_path: str | None,
) -> Path:
    if checkpoint_path is not None:
        return Path(checkpoint_path)

    return build_checkpoint_path(
        output_dir=config.checkpointing.output_dir,
        checkpoint_prefix=config.checkpointing.checkpoint_name_prefix,
        epoch=config.training.max_epochs,
    )


# ---------------------------------------------------------------------------
# _encode_prompt
#
# This function encodes prompt text into a batch-shaped token tensor.
# ---------------------------------------------------------------------------
def _encode_prompt(
    tokenizer: CharacterTokenizer,
    prompt: str,
    device: str,
) -> torch.Tensor:
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("prompt must be a non-empty string.")

    prompt_token_ids = tokenizer.encode(prompt)
    prompt_tensor = torch.tensor([prompt_token_ids], dtype=torch.long)

    return prompt_tensor.to(device)


# ---------------------------------------------------------------------------
# _decode_generated_output
#
# This function decodes one generated token sequence back into text.
# ---------------------------------------------------------------------------
def _decode_generated_output(
    tokenizer: CharacterTokenizer,
    generated_token_ids: torch.Tensor,
) -> str:
    token_id_list = generated_token_ids[0].tolist()

    return tokenizer.decode(token_id_list)


# ---------------------------------------------------------------------------
# run_generation
#
# This function wires the generation components together and prints generated
# output.
# ---------------------------------------------------------------------------
def run_generation(
    config_path: str,
    prompt: str,
    checkpoint_path: str | None = None,
) -> None:
    config = load_config(config_path)
    corpus_text = _read_text(
        dataset_path=config.data.dataset_path,
        encoding=config.data.encoding,
    )
    tokenizer = _build_tokenizer(corpus_text)
    vocab_size = _resolve_vocab_size(config, tokenizer)
    model = _build_model(config, vocab_size)
    optimizer = build_optimizer(model, config)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=config.training.device,
    )

    resolved_checkpoint_path = _resolve_checkpoint_path(
        config=config,
        checkpoint_path=checkpoint_path,
    )
    trainer.load_checkpoint(checkpoint_path=resolved_checkpoint_path)

    prompt_token_ids = _encode_prompt(
        tokenizer=tokenizer,
        prompt=prompt,
        device=config.training.device,
    )
    generated_token_ids = generate_next_tokens_greedy(
        model=model,
        prompt_token_ids=prompt_token_ids,
        max_new_tokens=config.generation.max_new_tokens,
    )
    generated_text = _decode_generated_output(
        tokenizer=tokenizer,
        generated_token_ids=generated_token_ids,
    )

    print(generated_text)


# ---------------------------------------------------------------------------
# main
#
# This function runs the generation script.
# ---------------------------------------------------------------------------
def main() -> None:
    args = _parse_args()

    run_generation(
        config_path=args.config,
        prompt=args.prompt,
        checkpoint_path=args.checkpoint,
    )


if __name__ == "__main__":
    main()
