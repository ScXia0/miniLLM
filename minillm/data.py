"""Small helpers for next-token language-model training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


def read_text(path: str | Path) -> str:
    """Read a UTF-8 text corpus from disk."""

    return Path(path).read_text(encoding="utf-8")


def split_train_val(ids: list[int], val_fraction: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
    """Split token ids into train and validation tensors."""

    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    n = len(ids)
    split = int(n * (1 - val_fraction))
    train_ids = torch.tensor(ids[:split], dtype=torch.long)
    val_ids = torch.tensor(ids[split:], dtype=torch.long)
    return train_ids, val_ids


@dataclass
class BatchConfig:
    """Shape information for random next-token batches."""

    batch_size: int
    block_size: int
    device: torch.device


def get_batch(data: torch.Tensor, cfg: BatchConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a batch of input/target sequences.

    For language modeling, the target is the input shifted one token to the left:

        x = [t0, t1, t2, ...]
        y = [t1, t2, t3, ...]

    This trains the model to predict the next token at every position.
    """

    if len(data) <= cfg.block_size:
        raise ValueError(
            f"dataset has {len(data)} tokens, but block_size={cfg.block_size}; "
            "use a smaller block_size or a larger corpus"
        )

    max_start = len(data) - cfg.block_size - 1
    starts = torch.randint(0, max_start + 1, (cfg.batch_size,))
    x = torch.stack([data[i : i + cfg.block_size] for i in starts])
    y = torch.stack([data[i + 1 : i + cfg.block_size + 1] for i in starts])
    return x.to(cfg.device), y.to(cfg.device)

