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


class HFTokenizerAdapter:
    """Wrap a Hugging Face tokenizer with the minojev tokenizer interface."""

    name = "hf"

    def __init__(self, tokenizer, source: str = "") -> None:
        self.tokenizer = tokenizer
        self.source = source
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        if pad is None:
            raise ValueError("HF tokenizer needs a pad or EOS token")
        self.pad_id = int(pad)
        self.eos_id = int(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else self.pad_id
        self.bos_id = int(tokenizer.bos_token_id) if tokenizer.bos_token_id is not None else self.eos_id
        self.vocab_size = len(tokenizer)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = list(self.tokenizer.encode(text, add_special_tokens=False))
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def to_config(self) -> dict:
        return {"name": self.name, "source": self.source, "vocab_size": self.vocab_size}

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(directory)


TOKENIZER_CONFIG_NAME = "minojev_tokenizer.json"


def save_tokenizer(tokenizer, directory: Path) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / TOKENIZER_CONFIG_NAME).write_text(
        json.dumps(tokenizer.to_config(), ensure_ascii=False, indent=2) + "\n"
    )
    if isinstance(tokenizer, HFTokenizerAdapter):
        tokenizer.save(directory)


def load_tokenizer(directory: Path):
    directory = Path(directory)
    path = directory / TOKENIZER_CONFIG_NAME
    if not path.exists() and (directory / "tokenizer.json").exists():
        path = directory / "tokenizer.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {TOKENIZER_CONFIG_NAME} under {directory}")
    config = json.loads(path.read_text())
    if config.get("name") == ByteTokenizer.name:
        return ByteTokenizer.from_config(config)
    if config.get("name") == HFTokenizerAdapter.name:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(directory)
        return HFTokenizerAdapter(tokenizer, source=config.get("source", ""))
    raise ValueError(f"Unsupported tokenizer config: {config}")
