"""Decision heads over candidate-path hidden states.

One shared head serves all three primitives:

- every candidate path gets a scalar logit from a shared linear scorer;
- choice questions pass their candidate set through a small set-attention
  block that mixes candidates within the question before the softmax, which
  keeps the readout permutation-equivariant and aware of the candidate count;
- boolean questions score a single proposition and return ``softmax([0, z])``
  (equivalently ``sigmoid(z)`` for the true candidate);
- score questions softmax their ordered level logits and report the
  probability-weighted expected level.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .types import BOOLEAN, CHOICE, SCORE


@dataclass
class GroupOutput:
    probabilities: torch.Tensor
    logits: torch.Tensor
    expected: torch.Tensor | None = None

    def to_dict(self, candidate_ids: list[str]) -> dict:
        record = {
            "candidate_ids": list(candidate_ids),
            "probabilities": [float(round(value, 8)) for value in self.probabilities.tolist()],
            "logits": [float(round(value, 8)) for value in self.logits.tolist()],
        }
        if self.expected is not None:
            record["expected"] = float(round(float(self.expected), 8))
            record["level"] = int(torch.argmax(self.probabilities).item())
        return record


class DecisionHead(nn.Module):
    def __init__(self, hidden_size: int, attention_dim: int = 128, num_heads: int = 4) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.scalar = nn.Linear(hidden_size, 1)
        self.set_input = nn.Linear(hidden_size + 1, attention_dim)
        self.set_attention = nn.MultiheadAttention(attention_dim, num_heads, dropout=0.0, batch_first=True)
        self.set_output = nn.Linear(attention_dim, 1)
        nn.init.normal_(self.scalar.weight, std=0.02)
        nn.init.zeros_(self.scalar.bias)
        nn.init.zeros_(self.set_output.weight)
        nn.init.zeros_(self.set_output.bias)

    def forward(self, hidden: torch.Tensor, groups, group_kinds: list[str]) -> list[GroupOutput]:
        normalized = self.norm(hidden)
        base = self.scalar(normalized).squeeze(-1).float()
        scores = base
        comparative = [index for index, kind in enumerate(group_kinds) if kind in (CHOICE, BOOLEAN)]
        if comparative:
            delta = self._set_correction(normalized, groups, comparative)
            scores = base.index_add(0, self._choice_path_indices(groups, comparative), delta)
        outputs = []
        for group, kind in zip(groups, group_kinds):
            logits = scores[group.path_indices]
            if kind == BOOLEAN:
                probabilities = torch.softmax(logits, dim=-1)
                outputs.append(GroupOutput(probabilities=probabilities, logits=logits))
            elif kind == SCORE:
                probabilities = torch.softmax(logits, dim=-1)
                indices = torch.arange(len(group.path_indices), device=hidden.device, dtype=probabilities.dtype)
                expected = torch.sum(indices * probabilities)
                outputs.append(GroupOutput(probabilities=probabilities, logits=logits, expected=expected))
            else:
                probabilities = torch.softmax(logits, dim=-1)
                outputs.append(GroupOutput(probabilities=probabilities, logits=logits))
        return outputs

    def _choice_path_indices(self, groups, choice_rows: list[int]) -> torch.Tensor:
        flat = []
        for row in choice_rows:
            flat.extend(groups[row].path_indices)
        return torch.tensor(flat, dtype=torch.long, device=self.scalar.weight.device)

    def _set_correction(self, hidden: torch.Tensor, groups, choice_rows: list[int]) -> torch.Tensor:
        device = hidden.device
        candidates = [groups[row].path_indices for row in choice_rows]
        width = max(len(paths) for paths in candidates)
        batch = len(candidates)
        pooled = hidden.new_zeros((batch, width, hidden.shape[-1]))
        valid = torch.zeros((batch, width), dtype=torch.bool, device=device)
        for row, paths in enumerate(candidates):
            pooled[row, : len(paths)] = hidden[paths]
            valid[row, : len(paths)] = True
        count = valid.sum(-1).float()
        size_feature = torch.where(valid, torch.log(count)[:, None], torch.zeros_like(count)[:, None])
        features = self.set_input(torch.cat([pooled, size_feature.unsqueeze(-1)], dim=-1))
        mixed, _ = self.set_attention(features, features, features, key_padding_mask=~valid, need_weights=False)
        delta = self.set_output(torch.tanh(features + mixed)).squeeze(-1).float()
        flat = []
        for row, paths in enumerate(candidates):
            flat.extend(paths)
        return delta[valid]


def group_loss(outputs: list[GroupOutput], groups, targets: list[torch.Tensor], objective: str) -> torch.Tensor:
    """Loss per question for ``ce``, ``gold``, or ``brier`` objectives."""
    losses = []
    for output, group, target in zip(outputs, groups, targets):
        probabilities = output.probabilities.clamp(min=1e-9)
        target = target.to(probabilities.device)
        if objective == "brier":
            losses.append(torch.sum((probabilities - target) ** 2))
        else:
            losses.append(-torch.sum(target * torch.log(probabilities)))
    if not losses:
        return torch.zeros((), device=next(iter(outputs)).probabilities.device if outputs else "cpu")
    return torch.stack(losses)
