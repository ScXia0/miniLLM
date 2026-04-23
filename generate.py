"""从训练好的 miniLLM checkpoint 中生成文本。

运行示例：
    python generate.py --prompt "To build" --max_new_tokens 200
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from minillm.model import GPTConfig, MiniGPT
from minillm.tokenizer import CharTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample from miniLLM")
    parser.add_argument("--checkpoint", type=str, default="out/model.pt")
    parser.add_argument("--tokenizer", type=str, default="out/tokenizer.json")
    parser.add_argument("--prompt", type=str, default="To build")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    return parser.parse_args()


def pick_device(name: str) -> torch.device:
    """选择推理设备；auto 会优先尝试 CUDA，再尝试 Apple Silicon 的 MPS。"""

    if name == "cuda" or (name == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    if name == "mps" or (name == "auto" and torch.backends.mps.is_available()):
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)

    # checkpoint 保存了模型权重和模型结构配置，生成时必须二者都恢复。
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = GPTConfig(**checkpoint["model_config"])
    model = MiniGPT(config).to(device)
    model.load_state_dict(checkpoint["model_state"])

    # tokenizer 必须和训练时一致，否则同一个 token id 会对应不同字符。
    tokenizer = CharTokenizer.load(Path(args.tokenizer))
    prompt_ids = tokenizer.encode(args.prompt)
    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    y = model.generate(
        x,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(tokenizer.decode(y[0].tolist()))


if __name__ == "__main__":
    main()
