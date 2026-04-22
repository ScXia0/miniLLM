"""A deliberately simple character-level tokenizer.

Real LLMs usually use BPE/Unigram tokenizers because they compress text much
better than raw characters. For a from-zero implementation, a char tokenizer is
perfect for the first milestone: it makes tokenization transparent, keeps the
model small, and lets us focus on the Transformer training loop.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CharTokenizer:
    """Map characters to integer token ids and back again.

    The tokenizer owns two lookup tables:
    - stoi: string/character -> integer id
    - itos: integer id -> string/character

    We reserve id 0 for unknown characters so generation prompts do not crash if
    the user types a character that was not present in the training corpus.
    """

    stoi: dict[str, int]
    itos: dict[int, str]
    unk_token: str = "<unk>"

    @classmethod
    def train(cls, text: str) -> "CharTokenizer":
        """Build a vocabulary from all unique characters in the corpus."""

        chars = sorted(set(text))
        stoi = {"<unk>": 0}
        for i, ch in enumerate(chars, start=1):
            stoi[ch] = i
        itos = {i: ch for ch, i in stoi.items()}
        return cls(stoi=stoi, itos=itos)

    @property
    def vocab_size(self) -> int:
        """Number of tokens the model must be able to predict."""

        return len(self.stoi)

    def encode(self, text: str) -> list[int]:
        """Turn text into token ids.

        Unknown characters are mapped to 0 instead of raising an error. This is
        friendly for demos, and it mirrors the idea of special fallback tokens in
        production tokenizers.
        """

        return [self.stoi.get(ch, 0) for ch in text]

    def decode(self, ids: list[int]) -> str:
        """Turn token ids back into text."""

        pieces = []
        for token_id in ids:
            piece = self.itos.get(int(token_id), self.unk_token)
            pieces.append("�" if piece == self.unk_token else piece)
        return "".join(pieces)

    def save(self, path: str | Path) -> None:
        """Persist the vocabulary as JSON so training and generation agree."""

        path = Path(path)
        payload = {
            "stoi": self.stoi,
            "unk_token": self.unk_token,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        """Load a tokenizer created by :meth:`save`."""

        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        stoi = {str(ch): int(i) for ch, i in payload["stoi"].items()}
        itos = {i: ch for ch, i in stoi.items()}
        return cls(stoi=stoi, itos=itos, unk_token=payload.get("unk_token", "<unk>"))

