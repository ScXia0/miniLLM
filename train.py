"""从零训练一个很小的 GPT/LLaMA 风格语言模型。

运行示例：
    python train.py --max_steps 500

这个脚本刻意保持短小，但覆盖完整训练流程：
1. 读取文本
2. 训练 tokenizer
3. 构造 next-token batch
4. 训练 MiniGPT
5. 评估验证集 loss
6. 保存 checkpoint 和 tokenizer
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
    """选择训练设备；如果没有 GPU/MPS，就回退到最通用的 CPU。"""

    if name == "cuda" or (name == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    if name == "mps" or (name == "auto" and torch.backends.mps.is_available()):
        return torch.device("mps")
    return torch.device("cpu")


def pick_autocast_dtype(device: torch.device, dtype_name: str) -> tuple[torch.dtype, bool]:
    """选择 automatic mixed precision 使用的 dtype。

    mixed precision（混合精度）是 LLM 训练常见优化：部分矩阵计算用 fp16/bf16，
    可以省显存、提吞吐；对数值更敏感的部分仍可保持 fp32。这里让 CPU 训练保持
    fp32，保证初学路径最稳定。
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
        # Apple Silicon 的 autocast 通常对 float16 支持更稳。
        return torch.float16, True
    return torch.float32, False


def autocast_context(device: torch.device, dtype: torch.dtype, enabled: bool) -> ContextManager:
    """返回 autocast 上下文；不开启混合精度时返回空操作上下文。"""

    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def build_optimizer(model: MiniGPT, learning_rate: float, weight_decay: float) -> torch.optim.Optimizer:
    """创建带 LLM 常见参数分组的 AdamW 优化器。

    weight decay 是权重衰减，用来抑制参数过大，起到正则化作用。大矩阵权重通常
    使用 weight decay；bias 和 norm 参数通常不使用。很多工业 LLM 训练 recipe
    都会这样分组，这是一个很常见的面试点。
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
    """把友好的架构名翻译成具体模型组件选择。"""

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
    """cosine learning rate schedule：前期学习率较高，后期按余弦曲线平滑衰减。"""

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
    """用多个随机 batch 估计 train/validation loss。"""

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
                # gradient accumulation（梯度累积）会把多个 microbatch 的梯度累加
                # 后再更新一次参数。这里除以累积步数，是为了让最终梯度尺度等价于
                # 直接使用一个更大的 batch。
                scaled_loss = loss / args.gradient_accumulation_steps

            micro_losses.append(loss.item())
            scaler.scale(scaled_loss).backward()

        # mixed precision 下，GradScaler 可能会放大梯度以避免 fp16 下溢。
        # 做 gradient clipping（梯度裁剪）前必须先 unscale，否则裁剪阈值会作用在
        # 被人为放大的梯度上。
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
