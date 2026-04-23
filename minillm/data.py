"""next-token 语言模型训练所需的小工具。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


def read_text(path: str | Path) -> str:
    """从磁盘读取 UTF-8 文本语料。"""

    return Path(path).read_text(encoding="utf-8")


def split_train_val(ids: list[int], val_fraction: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
    """把 token id 序列切成训练集和验证集。"""

    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    n = len(ids)
    split = int(n * (1 - val_fraction))
    train_ids = torch.tensor(ids[:split], dtype=torch.long)
    val_ids = torch.tensor(ids[split:], dtype=torch.long)
    return train_ids, val_ids


@dataclass
class BatchConfig:
    """随机采样 batch 时需要的形状和设备配置。"""

    batch_size: int
    block_size: int
    device: torch.device


def get_batch(data: torch.Tensor, cfg: BatchConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """随机采样一批输入序列 x 和目标序列 y。

    对 next-token prediction 来说，目标 y 就是输入 x 向左平移一个 token：

        x = [t0, t1, t2, ...]
        y = [t1, t2, t3, ...]

    这样模型在每个位置都要学习“根据当前位置及之前的上下文预测下一个 token”。
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
