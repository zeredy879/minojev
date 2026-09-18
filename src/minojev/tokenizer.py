"""Tokenizers used by minojev backbones.

``ByteTokenizer`` is a dependency-free, fully offline tokenizer used by the
built-in tiny backbone. Every UTF-8 byte is one token, so any request is
encodable without downloading an external vocabulary. Hugging Face tokenizers
are adapted lazily in :mod:`minojev.backbone`.
"""

from __future__ import annotations

import json
from pathlib import Path

PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
SPECIALS = (PAD, BOS, EOS)


class ByteTokenizer:
    """Reversible byte-level tokenizer with three reserved special tokens."""

    name = "byte"
    version = 1

    def __init__(self) -> None:
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self.vocab_size = 3 + 256
        self._special_ids = {PAD: self.pad_id, BOS: self.bos_id, EOS: self.eos_id}

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = [byte + 3 for byte in text.encode("utf-8")]
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        raw = bytes(value - 3 for value in ids if value >= 3)
        return raw.decode("utf-8", errors="replace")

    def to_config(self) -> dict:
        return {"name": self.name, "version": self.version, "vocab_size": self.vocab_size}

    @classmethod
    def from_config(cls, config: dict) -> "ByteTokenizer":
        if config.get("name") != cls.name or int(config.get("version", 0)) != cls.version:
            raise ValueError(f"Unsupported tokenizer config: {config}")
        return cls()


def save_tokenizer(tokenizer, directory: Path) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "tokenizer.json").write_text(json.dumps(tokenizer.to_config(), ensure_ascii=False, indent=2) + "\n")


def load_tokenizer(directory: Path):
    directory = Path(directory)
    path = directory / "tokenizer.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing tokenizer.json under {directory}")
    config = json.loads(path.read_text())
    if config.get("name") == ByteTokenizer.name:
        return ByteTokenizer.from_config(config)
    return HFTokenizer(config)
