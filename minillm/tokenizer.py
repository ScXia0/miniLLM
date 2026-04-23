"""一个刻意保持简单的字符级 tokenizer。

tokenizer 的作用是把文本切成 token，再把 token 映射成整数 id。模型本身只能
处理整数/张量，不能直接处理字符串。

真实 LLM 通常使用 BPE 或 Unigram tokenizer：

- BPE，Byte Pair Encoding，字节对编码：
  从很小的基本单位开始，比如字符或字节，不断统计语料中最常一起出现的相邻
  片段，并把它们合并成新的 token。例如 "l" + "o" 频繁出现，就可能合并成
  "lo"；再继续合并出 "low"、"lower" 这类子词。BPE 的直觉是：高频片段应该
  用一个 token 表示，从而减少序列长度。

- Unigram tokenizer，一元语言模型 tokenizer：
  先准备一个较大的候选子词表，然后给每个子词一个概率。分词时选择一组最可能
  生成当前文本的子词组合；训练时会逐步删除贡献较小的子词，留下更紧凑的词表。
  它的直觉是：分词本身也是一个概率模型，不只是贪心合并。

BPE 和 Unigram 都是 subword tokenizer（子词 tokenizer）：token 既可以是单个
字符，也可以是常见词片段，甚至是完整单词。它们比纯字符级 tokenizer 更能压缩
文本，序列更短，训练和推理都更省。

本项目当前用的是 char-level tokenizer（字符级 tokenizer）：每个出现过的字符
就是一个 token。它压缩率不高，但实现最透明，适合作为从 0 理解 LLM 的第一步：
你可以清楚看到“文本 -> 字符 -> 整数 id -> 模型输入”的全过程。

一个字符级 tokenizer 的示例：

假设训练语料里出现过这些字符：

    "h", "i", "!", "你", "好"

那么 tokenizer 会构建类似这样的词表：

    "<unk>" -> 0
    "!"     -> 1
    "h"     -> 2
    "i"     -> 3
    "你"    -> 4
    "好"    -> 5

注意：真实 id 取决于词表排序，这里只是示意。

encode 示例：

    "hi!"  -> ["h", "i", "!"]   -> [2, 3, 1]
    "你好" -> ["你", "好"]       -> [4, 5]

decode 示例：

    [2, 3, 1] -> ["h", "i", "!"] -> "hi!"

如果输入里有训练时没见过的字符，例如 "x"，它会被映射到 0，也就是 <unk>。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CharTokenizer:
    """把字符映射成整数 token id，也能把整数 id 还原成字符。

    这里维护两张查表：
    - stoi: string to integer，字符 -> 整数 id
    - itos: integer to string，整数 id -> 字符

    id 0 保留给未知字符。如果生成时 prompt 里出现训练语料没见过的字符，
    程序不会直接报错，而是映射到这个未知 token。
    """

    stoi: dict[str, int]
    itos: dict[int, str]
    unk_token: str = "<unk>"

    @classmethod
    def train(cls, text: str) -> "CharTokenizer":
        """从语料里出现过的所有字符构建词表。"""

        chars = sorted(set(text))
        stoi = {"<unk>": 0}
        for i, ch in enumerate(chars, start=1):
            stoi[ch] = i
        itos = {i: ch for ch, i in stoi.items()}
        return cls(stoi=stoi, itos=itos)

    @property
    def vocab_size(self) -> int:
        """词表大小，也就是模型最终分类头需要预测的类别数。"""

        return len(self.stoi)

    def encode(self, text: str) -> list[int]:
        """把文本转换成 token id 列表。

        未知字符会映射到 0，而不是直接抛异常。这对 demo 更友好，也对应了
        工业 tokenizer 里常见的特殊兜底 token。
        """

        return [self.stoi.get(ch, 0) for ch in text]

    def decode(self, ids: list[int]) -> str:
        """把 token id 列表还原成文本。"""

        pieces = []
        for token_id in ids:
            piece = self.itos.get(int(token_id), self.unk_token)
            pieces.append("�" if piece == self.unk_token else piece)
        return "".join(pieces)

    def save(self, path: str | Path) -> None:
        """把词表保存成 JSON，确保训练和生成使用同一套映射。"""

        path = Path(path)
        payload = {
            "stoi": self.stoi,
            "unk_token": self.unk_token,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        """加载由 save 方法保存的 tokenizer。"""

        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        stoi = {str(ch): int(i) for ch, i in payload["stoi"].items()}
        itos = {i: ch for ch, i in stoi.items()}
        return cls(stoi=stoi, itos=itos, unk_token=payload.get("unk_token", "<unk>"))
