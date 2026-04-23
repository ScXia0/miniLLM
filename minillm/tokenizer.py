"""一个刻意保持简单的字符级 tokenizer。

真实 LLM 通常使用 BPE 或 Unigram tokenizer，因为它们能把文本压缩成更短的
token 序列，训练和推理都更省。这里先用字符级 tokenizer，是为了让“文本如何
变成整数”完全透明，把注意力放在 Transformer 训练主流程上。
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
