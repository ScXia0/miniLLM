"""Train a tiny GPT-style language model from scratch.

Run:
    python train.py --max_steps 500

This script is intentionally compact but complete:
1. read text
2. train a tokenizer
3. create next-token batches
4. train MiniGPT
5. evaluate validation loss
6. save checkpoint + tokenizer
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from minillm.data import BatchConfig, get_batch, read_text, split_train_val
from minillm.model import GPTConfig, MiniGPT
from minillm.tokenizer import CharTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train miniLLM")
    parser.add_argument("--data_path", type=str, default="data/sample.txt")
    parser.add_argument("--out_dir", type=str, default="out")
    parser.add_argument("--block_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--n_embd", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--eval_iters", type=int, default=20)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    return parser.parse_args()


def pick_device(name: str) -> torch.device:
    """Select a device while keeping CPU as the portable fallback."""

    if name == "cuda" or (name == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    if name == "mps" or (name == "auto" and torch.backends.mps.is_available()):
        return torch.device("mps")
    return torch.device("cpu")


def cosine_lr(step: int, max_steps: int, base_lr: float) -> float:
    """A tiny cosine schedule: high learning rate early, gentle decay later."""

    progress = step / max(1, max_steps)
    return 0.1 * base_lr + 0.9 * base_lr * 0.5 * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def estimate_loss(
    model: MiniGPT,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    batch_cfg: BatchConfig,
    eval_iters: int,
) -> dict[str, float]:
    """Estimate train/validation loss from several random batches."""

    model.eval()
    losses = {}
    for split_name, data in [("train", train_data), ("val", val_data)]:
        split_losses = []
        for _ in range(eval_iters):
            x, y = get_batch(data, batch_cfg)
            _, loss = model(x, y)
            assert loss is not None
            split_losses.append(loss.item())
        losses[split_name] = sum(split_losses) / len(split_losses)
    model.train()
    return losses


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)

    text = read_text(args.data_path)
    tokenizer = CharTokenizer.train(text)
    ids = tokenizer.encode(text)
    train_data, val_data = split_train_val(ids, args.val_fraction)

    min_required = args.block_size + 2
    if len(train_data) < min_required or len(val_data) < min_required:
        raise ValueError(
            "training and validation splits must each be longer than block_size; "
            "use a larger corpus, smaller --block_size, or smaller --val_fraction"
        )

    tokenizer_path = out_dir / "tokenizer.json"
    tokenizer.save(tokenizer_path)

    model_cfg = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = MiniGPT(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    batch_cfg = BatchConfig(batch_size=args.batch_size, block_size=args.block_size, device=device)

    print(f"device: {device}")
    print(f"vocab_size: {tokenizer.vocab_size}")
    print(f"train tokens: {len(train_data)}, val tokens: {len(val_data)}")
    print(f"parameters: {model.num_parameters():,}")

    start_time = time.time()
    last_eval = {}
    for step in range(1, args.max_steps + 1):
        lr = cosine_lr(step, args.max_steps, args.learning_rate)
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = get_batch(train_data, batch_cfg)
        _, loss = model(x, y)
        assert loss is not None

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step == 1 or step % args.eval_interval == 0 or step == args.max_steps:
            last_eval = estimate_loss(model, train_data, val_data, batch_cfg, args.eval_iters)
            elapsed = time.time() - start_time
            print(
                f"step {step:5d}/{args.max_steps} | "
                f"train {last_eval['train']:.4f} | val {last_eval['val']:.4f} | "
                f"lr {lr:.2e} | {elapsed:.1f}s"
            )

    checkpoint = {
        "model_state": model.state_dict(),
        "model_config": model_cfg.__dict__,
        "tokenizer_path": str(tokenizer_path),
        "step": args.max_steps,
        "last_eval": last_eval,
    }
    ckpt_path = out_dir / "model.pt"
    torch.save(checkpoint, ckpt_path)

    metadata = {
        "checkpoint": str(ckpt_path),
        "tokenizer": str(tokenizer_path),
        "model_config": model_cfg.__dict__,
        "last_eval": last_eval,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
