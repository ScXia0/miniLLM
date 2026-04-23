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
from contextlib import nullcontext
import json
import math
import time
from pathlib import Path
from typing import ContextManager

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
    parser.add_argument("--architecture", type=str, default="gpt", choices=["gpt", "llama"])
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "float32", "float16", "bfloat16"])
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


def pick_autocast_dtype(device: torch.device, dtype_name: str) -> tuple[torch.dtype, bool]:
    """Choose the dtype used by automatic mixed precision.

    Mixed precision is a standard LLM training optimization: some matrix
    operations run in fp16/bf16 to save memory and improve throughput, while
    sensitive operations can remain in fp32. CPU training stays in fp32 here so
    the beginner path remains portable and predictable.
    """

    if dtype_name == "float32" or device.type == "cpu":
        return torch.float32, False
    if dtype_name == "float16":
        return torch.float16, True
    if dtype_name == "bfloat16":
        return torch.bfloat16, True
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16, True
    if device.type == "mps":
        # Apple Silicon generally handles float16 autocast better than bfloat16.
        return torch.float16, True
    return torch.float32, False


def autocast_context(device: torch.device, dtype: torch.dtype, enabled: bool) -> ContextManager:
    """Return an autocast context manager or a no-op context manager."""

    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def build_optimizer(model: MiniGPT, learning_rate: float, weight_decay: float) -> torch.optim.Optimizer:
    """Create AdamW with LLM-style weight-decay parameter groups.

    Large matrix weights benefit from weight decay, but biases and normalization
    parameters usually do not. Splitting them mirrors the optimizer setup used in
    many production LLM training recipes and is a useful interview detail.
    """

    decay_params = []
    no_decay_params = []
    for _, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(optim_groups, lr=learning_rate)


def architecture_options(name: str) -> dict[str, str | float]:
    """Translate a friendly architecture name into model component choices."""

    if name == "gpt":
        return {
            "norm_type": "layernorm",
            "mlp_type": "gelu",
            "position_embedding_type": "learned",
            "rope_theta": 10000.0,
        }
    if name == "llama":
        return {
            "norm_type": "rmsnorm",
            "mlp_type": "swiglu",
            "position_embedding_type": "rope",
            "rope_theta": 10000.0,
        }
    raise ValueError(f"unknown architecture: {name}")


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
    amp_dtype: torch.dtype,
    use_amp: bool,
) -> dict[str, float]:
    """Estimate train/validation loss from several random batches."""

    model.eval()
    losses = {}
    for split_name, data in [("train", train_data), ("val", val_data)]:
        split_losses = []
        for _ in range(eval_iters):
            x, y = get_batch(data, batch_cfg)
            with autocast_context(batch_cfg.device, amp_dtype, use_amp):
                _, loss = model(x, y)
            assert loss is not None
            split_losses.append(loss.item())
        losses[split_name] = sum(split_losses) / len(split_losses)
    model.train()
    return losses


def main() -> None:
    args = parse_args()
    if args.gradient_accumulation_steps < 1:
        raise ValueError("--gradient_accumulation_steps must be >= 1")

    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    amp_dtype, use_amp = pick_autocast_dtype(device, args.dtype)

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

    arch_options = architecture_options(args.architecture)
    model_cfg = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        **arch_options,
    )
    model = MiniGPT(model_cfg).to(device)
    optimizer = build_optimizer(model, args.learning_rate, args.weight_decay)
    batch_cfg = BatchConfig(batch_size=args.batch_size, block_size=args.block_size, device=device)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and amp_dtype == torch.float16))
    effective_batch_tokens = args.batch_size * args.block_size * args.gradient_accumulation_steps

    print(f"device: {device}")
    print(f"architecture: {args.architecture}")
    print(
        "components: "
        f"{model_cfg.norm_type}, {model_cfg.mlp_type}, {model_cfg.position_embedding_type}"
    )
    print(f"mixed precision: {'on' if use_amp else 'off'} ({amp_dtype})")
    print(f"vocab_size: {tokenizer.vocab_size}")
    print(f"train tokens: {len(train_data)}, val tokens: {len(val_data)}")
    print(f"parameters: {model.num_parameters():,}")
    print(f"gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"effective batch tokens: {effective_batch_tokens:,}")

    start_time = time.time()
    last_log_time = start_time
    last_log_step = 0
    last_eval = {}
    for step in range(1, args.max_steps + 1):
        lr = cosine_lr(step, args.max_steps, args.learning_rate)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        micro_losses = []
        for micro_step in range(args.gradient_accumulation_steps):
            x, y = get_batch(train_data, batch_cfg)
            with autocast_context(device, amp_dtype, use_amp):
                _, loss = model(x, y)
                assert loss is not None
                # Dividing by accumulation steps keeps the final gradient scale
                # equivalent to using one large batch instead of many microbatches.
                scaled_loss = loss / args.gradient_accumulation_steps

            micro_losses.append(loss.item())
            scaler.scale(scaled_loss).backward()

        # Gradients must be unscaled before clipping, otherwise the clipping
        # threshold would be applied to artificially enlarged fp16 gradients.
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        if step == 1 or step % args.eval_interval == 0 or step == args.max_steps:
            last_eval = estimate_loss(model, train_data, val_data, batch_cfg, args.eval_iters, amp_dtype, use_amp)
            now = time.time()
            elapsed = now - start_time
            log_elapsed = max(now - last_log_time, 1e-9)
            interval_steps = step - last_log_step
            tokens_per_second = interval_steps * effective_batch_tokens / log_elapsed
            last_log_time = now
            last_log_step = step
            print(
                f"step {step:5d}/{args.max_steps} | "
                f"train {last_eval['train']:.4f} | val {last_eval['val']:.4f} | "
                f"micro {sum(micro_losses) / len(micro_losses):.4f} | "
                f"lr {lr:.2e} | {tokens_per_second:,.0f} tok/s | {elapsed:.1f}s"
            )

    checkpoint = {
        "model_state": model.state_dict(),
        "model_config": model_cfg.__dict__,
        "tokenizer_path": str(tokenizer_path),
        "step": args.max_steps,
        "last_eval": last_eval,
        "train_config": {
            "batch_size": args.batch_size,
            "block_size": args.block_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_batch_tokens": effective_batch_tokens,
            "dtype": args.dtype,
            "amp_dtype": str(amp_dtype),
            "use_amp": use_amp,
            "architecture": args.architecture,
        },
    }
    ckpt_path = out_dir / "model.pt"
    torch.save(checkpoint, ckpt_path)

    metadata = {
        "checkpoint": str(ckpt_path),
        "tokenizer": str(tokenizer_path),
        "model_config": model_cfg.__dict__,
        "last_eval": last_eval,
        "train_config": checkpoint["train_config"],
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
