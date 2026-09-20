"""Backbones that turn token ids into hidden states.

Two options are provided:

- :class:`TinyLM`: a small decoder-only transformer with a native KV cache.
  It is fully self-contained, trains from scratch on CPU in minutes, and is
  what the test suite and the bundled reproduction use.
- :class:`HFBackbone`: a thin adapter over any Hugging Face causal LM,
  imported lazily so the core package has no hard dependency on transformers.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn


@dataclass
class TinyConfig:
    vocab_size: int = 259
    hidden_size: int = 96
    num_layers: int = 3
    num_heads: int = 4
    intermediate_size: int = 384
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if (self.hidden_size // self.num_heads) % 2:
            raise ValueError("head dimension must be even for rotary positions")

    @classmethod
    def from_dict(cls, data: dict) -> "TinyConfig":
        known = {key: value for key, value in data.items() if key in cls.__dataclass_fields__}
        return cls(**known)


def apply_rotary(hidden: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
    """Rotary position embedding over the last dimension of [B, H, T, D]."""
    half = hidden.shape[-1] // 2
    frequencies = 1.0 / (10000.0 ** (torch.arange(half, device=hidden.device).float() / half))
    angles = position_ids.float()[:, None, :, None] * frequencies[None, None, None, :]
    cosine, sine = angles.cos().to(hidden.dtype), angles.sin().to(hidden.dtype)
    left, right = hidden[..., :half], hidden[..., half:]
    return torch.cat([left * cosine - right * sine, left * sine + right * cosine], dim=-1)


class TinyKVCache:
    """Key/value cache with batch replication for branching decisions."""

    def __init__(self, layers: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
        self.layers = layers

    @property
    def seq_length(self) -> int:
        return 0 if not self.layers else self.layers[0][0].shape[2]

    @property
    def batch_size(self) -> int:
        return 0 if not self.layers else self.layers[0][0].shape[0]

    def expand(self, indices: torch.Tensor) -> "TinyKVCache":
        return TinyKVCache([(key.index_select(0, indices), value.index_select(0, indices)) for key, value in self.layers])


class TinyAttention(nn.Module):
    def __init__(self, config: TinyConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.dropout = config.dropout

    def forward(self, hidden, attention_bias, position_ids, past=None):
        batch, length, _ = hidden.shape
        shape = (batch, length, self.num_heads, self.head_dim)
        query = apply_rotary(self.q_proj(hidden).view(shape).transpose(1, 2), position_ids)
        key = apply_rotary(self.k_proj(hidden).view(shape).transpose(1, 2), position_ids)
        value = self.v_proj(hidden).view(shape).transpose(1, 2)
        if past is not None:
            past_key, past_value = past
            key = torch.cat([past_key, key], dim=2)
            value = torch.cat([past_value, value], dim=2)
        scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim) + attention_bias
        weights = torch.softmax(scores.float(), dim=-1).to(hidden.dtype)
        if self.training and self.dropout:
            weights = torch.dropout(weights, self.dropout, self.training)
        output = (weights @ value).transpose(1, 2).reshape(batch, length, -1)
        return self.o_proj(output), key, value


class TinyBlock(nn.Module):
    def __init__(self, config: TinyConfig) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(config.hidden_size)
        self.attention = TinyAttention(config)
        self.post_norm = nn.LayerNorm(config.hidden_size)
        self.gate = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden, attention_bias, position_ids, past=None):
        residual = hidden
        hidden = self.input_norm(hidden)
        attention, key, value = self.attention(hidden, attention_bias, position_ids, past)
        hidden = residual + attention
        residual = hidden
        hidden = self.post_norm(hidden)
        hidden = self.down(nn.functional.silu(self.gate(hidden)) * self.up(hidden))
        return residual + hidden, key, value


class TinyLM(nn.Module):
    """Small decoder-only transformer exposing hidden states and logits."""

    supports_reuse = True

    def __init__(self, config: TinyConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList(TinyBlock(config) for _ in range(config.num_layers))
        self.final_norm = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    @property
    def hidden_size(self) -> int:
        return self.config.hidden_size

    def _attention_bias(self, attention_mask, query_positions, key_length, dtype):
        batch, length = query_positions.shape
        key_positions = torch.arange(key_length, device=query_positions.device)
        valid = key_positions[None, :] <= query_positions[:, :, None]
        if attention_mask is not None:
            valid = valid & attention_mask[:, None, :].bool()
        bias = torch.zeros(batch, 1, length, key_length, dtype=dtype, device=query_positions.device)
        return bias.masked_fill(~valid[:, None, :, :], torch.finfo(dtype).min)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: TinyKVCache | None = None,
        position_ids: torch.Tensor | None = None,
    ):
        batch, length = input_ids.shape
        past_length = past_key_values.seq_length if past_key_values is not None else 0
        if position_ids is None:
            position_ids = torch.arange(past_length, past_length + length, device=input_ids.device)[None, :].expand(batch, -1)
        hidden = self.embed_tokens(input_ids)
        if attention_mask is None:
            attention_mask = input_ids.new_ones(batch, past_length + length)
        attention_bias = self._attention_bias(attention_mask, position_ids, past_length + length, hidden.dtype)
        cache: list[tuple[torch.Tensor, torch.Tensor]] = []
        for index, block in enumerate(self.blocks):
            past = None if past_key_values is None else past_key_values.layers[index]
            hidden, key, value = block(hidden, attention_bias, position_ids, past)
            cache.append((key, value))
        hidden = self.final_norm(hidden)
        return hidden, TinyKVCache(cache)

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden)

    def to_config(self) -> dict:
        return {"type": "tiny", **asdict(self.config)}

    def save(self, directory: Path) -> None:
        from safetensors.torch import save_file

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        state = {key: value.contiguous() for key, value in self.state_dict().items() if key != "lm_head.weight"}
        save_file(state, str(directory / "model.safetensors"))


class HFBackbone:
    """Adapter over a Hugging Face causal language model.

    Hidden states come from the base model (``model.model``), keeping the
    forward pass free of the vocabulary projection. The causal LM head is only
    used by the native-logits engine.
    """

    supports_reuse = False

    def __init__(self, model, tokenizer=None, source: str = "") -> None:
        self.model = model
        self.base = getattr(model, "model", None) or model.get_decoder()
        self.tokenizer = tokenizer
        self.source = source
        self.hidden_size = int(model.config.hidden_size)

    def refresh_base(self) -> None:
        self.base = getattr(self.model, "model", None) or self.model.get_decoder()

    def __call__(self, input_ids, attention_mask=None, past_key_values=None, position_ids=None, use_cache=False):
        kwargs = {"input_ids": input_ids, "use_cache": use_cache, "return_dict": True}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        if past_key_values is not None:
            kwargs["past_key_values"] = past_key_values
        if position_ids is not None:
            kwargs["position_ids"] = position_ids
        output = self.base(**kwargs)
        return output.last_hidden_state, output.past_key_values

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.model.get_output_embeddings()(hidden)

    def parameters(self):
        return self.model.parameters()

    def named_parameters(self):
        return self.model.named_parameters()

    def to(self, device):
        self.model.to(device)
        return self

    def train(self, mode: bool = True):
        self.model.train(mode)
        return self

    def eval(self):
        self.model.eval()
        return self

    def save(self, directory) -> None:
        self.model.save_pretrained(directory)

    def to_config(self) -> dict:
        return {"type": "hf", "source": self.source, "hidden_size": self.hidden_size}


def _is_adapter_directory(path: str) -> bool:
    try:
        return (Path(path) / "adapter_config.json").exists()
    except (OSError, TypeError):
        return False


def load_hf_backbone(name_or_path: str, dtype: str = "float32", device: str = "auto", for_training: bool = False):
    import transformers

    from .tokenizer import HFTokenizerAdapter

    dtype_map = {
        "auto": "auto",
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    if dtype not in dtype_map:
        raise ValueError(f"Unsupported dtype {dtype!r}")
    if _is_adapter_directory(name_or_path):
        adapter_config = json.loads((Path(name_or_path) / "adapter_config.json").read_text())
        base_source = adapter_config.get("base_model_name_or_path", "Qwen/Qwen3-0.6B-Base")
        base = transformers.AutoModelForCausalLM.from_pretrained(base_source, dtype=dtype_map[dtype])
        from peft import PeftModel

        model = PeftModel.from_pretrained(base, name_or_path)
        tokenizer_source = base_source
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(name_or_path, dtype=dtype_map[dtype])
        tokenizer_source = name_or_path
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_source)
    except (OSError, ValueError):
        tokenizer = None
    if tokenizer is not None and tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    adapter = HFTokenizerAdapter(tokenizer, source=tokenizer_source) if tokenizer is not None else None
    if adapter is None:
        from .tokenizer import load_tokenizer

        try:
            adapter = load_tokenizer(Path(tokenizer_source))
        except (FileNotFoundError, ValueError):
            adapter = None
    model.to(resolve_device(device))
    model.train() if for_training else model.eval()
    return HFBackbone(model, tokenizer=adapter, source=name_or_path)


def resolve_device(device: str = "auto") -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
