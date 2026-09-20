"""Decision model: encoding, forward passes, scoring, and checkpoints.

The model never decodes output tokens. One forward pass turns every candidate
path into a hidden state; the decision heads turn those hidden states into
complete distributions.

Two serving modes are provided:

- ``fresh``: every path is padded and scored in one batch.
- ``reuse``: each distinct state is prefilled once, its cache is replicated
  across branches, and the question/candidate suffixes are scored together.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from .backbone import TinyConfig, TinyLM, resolve_device
from .encoding import EncodedBatch, encode_requests, pad_paths, pad_sequences
from .heads import DecisionHead, GroupOutput
from .tokenizer import ByteTokenizer, load_tokenizer, save_tokenizer
from .types import BOOLEAN, CHOICE, SCORE, Question, Request

TRUE_TOKENS = (True, 1, "true", "True", "yes", "Yes")
FALSE_TOKENS = (False, 0, "false", "False", "no", "No")


@dataclass
class ScoreOptions:
    mode: str = "fresh"
    batch_requests: int = 16
    device: str = "auto"


class DecisionModel(torch.nn.Module):
    def __init__(
        self,
        backbone,
        tokenizer,
        head: DecisionHead | None = None,
        config: dict | None = None,
        calibration: "Calibration | None" = None,
    ) -> None:
        super().__init__()
        from .calibrate import Calibration

        self.backbone = backbone
        self.tokenizer = tokenizer
        hidden = int(getattr(backbone, "hidden_size"))
        self.head = head if head is not None else DecisionHead(hidden)
        self.config = config or {}
        self.calibration = calibration or Calibration()

    @classmethod
    def new(cls, tokenizer: ByteTokenizer | None = None, **overrides) -> "DecisionModel":
        tokenizer = tokenizer or ByteTokenizer()
        config = TinyConfig(vocab_size=tokenizer.vocab_size, **overrides)
        return cls(TinyLM(config), tokenizer, config={"backbone": {"type": "tiny"}})

    def encode(self, requests: list[Request]) -> EncodedBatch:
        return encode_requests(requests, self.tokenizer)

    def backbone_parameters(self):
        return list(self.backbone.parameters())

    def trainable_parameters(self):
        if isinstance(self.backbone, torch.nn.Module):
            return list(self.parameters())
        return list(self.head.parameters()) + list(self.backbone.parameters())

    def to(self, device) -> "DecisionModel":
        self.backbone.to(device)
        self.head.to(device)
        return self

    def eval_mode(self, device: str = "auto") -> "DecisionModel":
        self.to(resolve_device(device))
        self.backbone.eval()
        self.head.eval()
        return self

    def train_mode(self) -> "DecisionModel":
        self.backbone.train()
        self.head.train()
        return self

    def forward_paths(self, encoded: EncodedBatch):
        device = next(self.head.parameters()).device
        tokens, mask = pad_paths(encoded.paths, self.tokenizer.pad_id, device)
        hidden, _ = self.backbone(input_ids=tokens, attention_mask=mask)
        lengths = mask.sum(-1) - 1
        rows = hidden[torch.arange(len(encoded.paths), device=device), lengths]
        return rows.float()

    def forward_reuse(self, encoded: EncodedBatch):
        device = next(self.head.parameters()).device
        state_order: list[str] = []
        state_to_paths: dict[str, list[int]] = {}
        for index, path in enumerate(encoded.paths):
            key = encoded.state_keys[path.request_index]
            if key not in state_to_paths:
                state_to_paths[key] = []
                state_order.append(key)
            state_to_paths[key].append(index)
        state_sequences = [encoded.paths[state_to_paths[key][0]].state_ids for key in state_order]
        hidden_rows: list[torch.Tensor | None] = [None] * len(encoded.paths)
        for state_index, key in enumerate(state_order):
            path_indices = state_to_paths[key]
            count = len(path_indices)
            prefix_length = len(state_sequences[state_index])
            prefix_ids = torch.tensor([state_sequences[state_index]], dtype=torch.long, device=device)
            prefix_mask = torch.ones((1, prefix_length), dtype=torch.long, device=device)
            _, cache = self.backbone(input_ids=prefix_ids, attention_mask=prefix_mask)
            branch = torch.zeros((count,), dtype=torch.long, device=device)
            branch_cache = cache.expand(branch)
            suffixes = [encoded.paths[index].suffix_ids for index in path_indices]
            suffix_tokens, suffix_mask = pad_sequences(suffixes, self.tokenizer.pad_id, device)
            full_mask = torch.cat([prefix_mask.expand(count, -1), suffix_mask], dim=1)
            positions = prefix_length + torch.arange(suffix_tokens.shape[1], device=device)[None, :].expand(count, -1)
            hidden, _ = self.backbone(
                input_ids=suffix_tokens,
                attention_mask=full_mask,
                past_key_values=branch_cache,
                position_ids=positions,
            )
            lengths = suffix_mask.sum(-1) - 1
            rows = hidden[torch.arange(count, device=device), lengths]
            for row, path_index in enumerate(path_indices):
                hidden_rows[path_index] = rows[row].float()
        return torch.stack(hidden_rows)

    def run_encoded(self, encoded: EncodedBatch, mode: str = "fresh") -> list[GroupOutput]:
        if mode == "reuse":
            if not getattr(self.backbone, "supports_reuse", False):
                raise ValueError("This backbone does not support shared-prefix reuse; use mode='fresh'")
            hidden = self.forward_reuse(encoded)
        elif mode == "fresh":
            hidden = self.forward_paths(encoded)
        else:
            raise ValueError(f"Unknown mode {mode!r}")
        return self.head(hidden, encoded.groups, [group.kind for group in encoded.groups])

    def score(self, requests: list[Request], options: ScoreOptions | None = None) -> list[dict]:
        from .calibrate import apply_calibration

        options = options or ScoreOptions()
        self.eval_mode(options.device)
        records: list[dict] = []
        chunk = max(1, options.batch_requests)
        with torch.no_grad():
            for start in range(0, len(requests), chunk):
                subset = requests[start : start + chunk]
                encoded = self.encode(subset)
                began = time.perf_counter()
                outputs = self.run_encoded(encoded, options.mode)
                outputs = [
                    apply_calibration(output, group.kind, self.calibration)
                    for output, group in zip(outputs, encoded.groups)
                ]
                elapsed = time.perf_counter() - began
                records.extend(self._records(encoded, outputs, options.mode, elapsed))
        return records

    def _records(self, encoded: EncodedBatch, outputs: list[GroupOutput], mode: str, elapsed: float) -> list[dict]:
        records = []
        for group, output in zip(encoded.groups, outputs):
            request = encoded.requests[group.request_index]
            question = request.questions[group.question_index]
            record = {
                "id": request.id,
                "qid": question.qid,
                "type": question.kind,
                "state": request.state,
                "instructions": question.instructions,
                **output.to_dict(group.candidate_ids),
                "decode_steps": 0,
                "mode": mode,
                "forward_seconds": round(elapsed, 8),
            }
            if question.family:
                record["family"] = question.family
            record["input_tokens"] = sum(len(encoded.paths[index].token_ids) for index in group.path_indices)
            if question.qid in request.gold:
                gold = request.gold[question.qid]
                record["gold"] = gold
                record["correct"] = is_correct(question, output, gold)
            if question.qid in request.teacher:
                record["teacher"] = request.teacher[question.qid]
            records.append(record)
        return records

    def save(self, directory: str | Path, save_backbone: bool = True) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        backbone_config = self.backbone.to_config()
        if backbone_config.get("type") == "hf":
            if save_backbone:
                self.backbone.save(directory / "backbone")
                save_tokenizer(self.tokenizer, directory / "backbone")
        else:
            self.backbone.save(directory)
            save_tokenizer(self.tokenizer, directory)
        from safetensors.torch import save_file

        save_file(
            {key: value.contiguous() for key, value in self.head.state_dict().items()},
            str(directory / "head.safetensors"),
        )
        config = {
            "format": "minojev-checkpoint",
            "version": 2,
            "backbone": backbone_config,
            "head": {
                "hidden_size": self.head.scalar.in_features,
                "attention_dim": self.head.set_attention.embed_dim,
                "num_heads": self.head.set_attention.num_heads,
            },
            "objective": self.config.get("objective"),
            "metrics": self.config.get("metrics"),
            "calibration": self.calibration.to_dict(),
        }
        (directory / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")

    @classmethod
    def load(cls, directory: str | Path, device: str = "auto") -> "DecisionModel":
        from safetensors.torch import load_file

        directory = Path(directory)
        config = json.loads((directory / "config.json").read_text())
        if config.get("format") != "minojev-checkpoint":
            raise ValueError(f"Not a minojev checkpoint: {directory}")
        backbone_entry = config["backbone"]
        if backbone_entry.get("type") == "hf":
            from .backbone import load_hf_backbone

            backbone_path = directory / "backbone"
            source = str(backbone_path) if backbone_path.exists() else backbone_entry.get("source")
            backbone = load_hf_backbone(
                source,
                dtype=backbone_entry.get("dtype", "float32"),
                device=device,
            )
            if backbone_path.exists() and (backbone_path / "minojev_tokenizer.json").exists():
                tokenizer = load_tokenizer(backbone_path)
            elif backbone.tokenizer is not None:
                tokenizer = backbone.tokenizer
            else:
                tokenizer = load_tokenizer(directory)
            hidden_size = int(backbone_entry.get("hidden_size", backbone.hidden_size))
        else:
            tokenizer = load_tokenizer(directory)
            backbone_config = TinyConfig.from_dict(backbone_entry)
            backbone = TinyLM(backbone_config)
            missing, unexpected = backbone.load_state_dict(load_file(str(directory / "model.safetensors")), strict=False)
            if set(missing) - {"lm_head.weight"} or unexpected:
                raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
            hidden_size = backbone_config.hidden_size
        head_config = config.get("head", {})
        head = DecisionHead(
            hidden_size=int(head_config.get("hidden_size", hidden_size)),
            attention_dim=int(head_config.get("attention_dim", 128)),
            num_heads=int(head_config.get("num_heads", 4)),
        )
        head.load_state_dict(load_file(str(directory / "head.safetensors")))
        from .calibrate import Calibration

        model = cls(backbone, tokenizer, head, config=config, calibration=Calibration.from_dict(config.get("calibration")))
        model.eval_mode(device)
        return model


def gold_target(question: Question, gold, size: int) -> torch.Tensor | None:
    if gold is None:
        return None
    if question.kind == BOOLEAN:
        index = 1 if gold in TRUE_TOKENS else 0
    elif question.kind == SCORE:
        index = int(gold)
    else:
        if gold not in question.candidate_ids:
            raise ValueError(f"gold {gold!r} is not a candidate of {question.qid}")
        index = question.candidate_ids.index(gold)
    target = torch.zeros(size)
    target[index] = 1.0
    return target


def is_correct(question: Question, output: GroupOutput, gold) -> bool:
    if question.kind == BOOLEAN:
        predicted = bool(output.probabilities[1] >= output.probabilities[0])
        expected = gold in TRUE_TOKENS
        return predicted == expected
    if question.kind == SCORE:
        return int(torch.argmax(output.probabilities).item()) == int(gold)
    probabilities = output.probabilities.tolist()
    best = max(range(len(probabilities)), key=probabilities.__getitem__)
    if gold not in question.candidate_ids:
        raise ValueError(f"gold {gold!r} is not a candidate of {question.qid}")
    return best == question.candidate_ids.index(gold)
