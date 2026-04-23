"""教学用 tokenizer 实现：保留字符级 CharTokenizer，并新增 BPE tokenizer。

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

本项目保留了 char-level tokenizer（字符级 tokenizer）：每个出现过的字符就是
一个 token。它压缩率不高，但实现最透明，适合作为从 0 理解 LLM 的第一步：
你可以清楚看到“文本 -> 字符 -> 整数 id -> 模型输入”的全过程。

同时，本项目也新增了 BPETokenizer。你可以通过 `train.py --tokenizer char`
和 `train.py --tokenizer bpe` 对比两种实现。

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

from collections import Counter
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
            "tokenizer_type": "char",
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


def _merge_pair_once(tokens: list[str], left: str, right: str, merged: str) -> list[str]:
    """把 token 序列里所有相邻的 (left, right) 合并成 merged。

    这是 BPE 的核心操作：如果当前序列中经常出现 "l", "o" 这对相邻 token，
    就把它们替换成一个新 token "lo"。
    """

    out = []
    i = 0
    while i < len(tokens):
        if i + 1 < len(tokens) and tokens[i] == left and tokens[i + 1] == right:
            out.append(merged)
            i += 2
        else:
            out.append(tokens[i])
            i += 1
    return out


@dataclass
class BPETokenizer:
    """一个极简 BPE tokenizer，用来和 CharTokenizer 对比实现差异。

    BPE，Byte Pair Encoding，字节对编码/字符对编码，在这里从字符级 token
    开始训练：

    1. 初始状态：每个字符都是一个 token
    2. 统计当前 token 序列里最常见的相邻 token pair
    3. 把这个 pair 合并成一个新 token
    4. 重复这个过程，直到达到目标词表大小，或没有值得合并的高频 pair

    一个非常小的示例：

        训练文本: "low lower lowest"

        初始 token:
            ["l", "o", "w", " ", "l", "o", "w", "e", "r", ...]

        如果 ("l", "o") 最常出现，就合并成 "lo":
            ["lo", "w", " ", "lo", "w", "e", "r", ...]

        如果 ("lo", "w") 又很常出现，就合并成 "low":
            ["low", " ", "low", "e", "r", ...]

        最终 encode("lower") 可能变成：
            ["low", "e", "r"] -> [某个 id, 某个 id, 某个 id]

    和 CharTokenizer 的区别：
    - CharTokenizer: "lower" -> ["l", "o", "w", "e", "r"]，一定是 5 个 token
    - BPETokenizer:  "lower" -> ["low", "e", "r"] 或 ["low", "er"]，token 更少

    这里实现的是教学版 BPE：为了让算法清楚，直接在字符序列上做 pair merge。
    工业级 BPE 通常会做 byte-level 处理、特殊 token、正则预分词等更多工程细节。
    """

    stoi: dict[str, int]
    itos: dict[int, str]
    merges: list[tuple[str, str]]
    unk_token: str = "<unk>"

    @classmethod
    def train(cls, text: str, vocab_size: int = 256, min_pair_freq: int = 2) -> "BPETokenizer":
        """从文本训练 BPE 词表和 merge 规则。

        vocab_size 是目标词表大小。真实训练中词表可能是 32k、50k、100k；这里默认
        256 是为了适合小语料和教学演示。
        """

        chars = sorted(set(text))
        stoi = {"<unk>": 0}
        for i, ch in enumerate(chars, start=1):
            stoi[ch] = i

        tokens = list(text)
        merges: list[tuple[str, str]] = []
        target_vocab_size = max(vocab_size, len(stoi))

        while len(stoi) < target_vocab_size:
            pair_counts = Counter(zip(tokens, tokens[1:]))
            if not pair_counts:
                break

            (left, right), freq = pair_counts.most_common(1)[0]
            if freq < min_pair_freq:
                break

            merged = left + right
            if merged in stoi:
                break

            stoi[merged] = len(stoi)
            merges.append((left, right))
            tokens = _merge_pair_once(tokens, left, right, merged)

        itos = {i: token for token, i in stoi.items()}
        return cls(stoi=stoi, itos=itos, merges=merges)

    @property
    def vocab_size(self) -> int:
        """词表大小，也就是模型最终分类头需要预测的类别数。"""

        return len(self.stoi)

    def encode(self, text: str) -> list[int]:
        """把文本转换成 BPE token id。

        编码时必须按训练阶段学到的 merge 顺序依次合并。这个顺序很重要：先合并
        ("l", "o") 得到 "lo"，后面才有机会继续合并 ("lo", "w") 得到 "low"。
        """

        tokens = [ch if ch in self.stoi else self.unk_token for ch in text]
        for left, right in self.merges:
            merged = left + right
            tokens = _merge_pair_once(tokens, left, right, merged)
        return [self.stoi.get(token, 0) for token in tokens]

    def decode(self, ids: list[int]) -> str:
        """把 BPE token id 还原成文本。"""

        pieces = []
        for token_id in ids:
            piece = self.itos.get(int(token_id), self.unk_token)
            pieces.append("�" if piece == self.unk_token else piece)
        return "".join(pieces)

    def save(self, path: str | Path) -> None:
        """保存 BPE 词表和 merge 规则。"""

        path = Path(path)
        payload = {
            "tokenizer_type": "bpe",
            "stoi": self.stoi,
            "merges": self.merges,
            "unk_token": self.unk_token,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        """加载由 save 方法保存的 BPE tokenizer。"""

        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        stoi = {str(token): int(i) for token, i in payload["stoi"].items()}
        itos = {i: token for token, i in stoi.items()}
        merges = [tuple(pair) for pair in payload.get("merges", [])]
        return cls(stoi=stoi, itos=itos, merges=merges, unk_token=payload.get("unk_token", "<unk>"))


def train_tokenizer(text: str, tokenizer_type: str, bpe_vocab_size: int = 256) -> CharTokenizer | BPETokenizer:
    """根据名称训练 tokenizer，供 train.py 调用。"""

    if tokenizer_type == "char":
        return CharTokenizer.train(text)
    if tokenizer_type == "bpe":
        return BPETokenizer.train(text, vocab_size=bpe_vocab_size)
    raise ValueError(f"unknown tokenizer_type: {tokenizer_type}")


def load_tokenizer(path: str | Path) -> CharTokenizer | BPETokenizer:
    """根据 JSON 里的 tokenizer_type 自动加载 CharTokenizer 或 BPETokenizer。

    早期 checkpoint 里没有 tokenizer_type 字段，因此默认按 char 读取，保持兼容。
    """

    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    tokenizer_type = payload.get("tokenizer_type", "char")
    if tokenizer_type == "char":
        return CharTokenizer.load(path)
    if tokenizer_type == "bpe":
        return BPETokenizer.load(path)
    raise ValueError(f"unknown tokenizer_type in {path}: {tokenizer_type}")
