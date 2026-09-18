"""Probability calibration for decision heads.

Temperature scaling per primitive: a single positive scalar divided into the
candidate logits before the softmax. Temperatures are fitted on a dev set by
minimizing negative log-likelihood against teacher distributions (falling back
to gold labels), and stored in the checkpoint so serving applies them
automatically.

Calibration never changes the predicted candidate: dividing logits by a
positive scalar preserves the argmax.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from .heads import GroupOutput
from .model import gold_target
from .types import BOOLEAN, CHOICE, SCORE, Request, teacher_distribution


@dataclass
class Calibration:
    temperatures: dict[str, float] = field(default_factory=lambda: {CHOICE: 1.0, BOOLEAN: 1.0, SCORE: 1.0})

    @classmethod
    def from_dict(cls, data: dict | None) -> "Calibration":
        temperatures = {CHOICE: 1.0, BOOLEAN: 1.0, SCORE: 1.0}
        for key, value in (data or {}).get("temperatures", {}).items():
            if key in temperatures:
                temperatures[key] = max(float(value), 1e-3)
        return cls(temperatures=temperatures)

    def to_dict(self) -> dict:
        return {"temperatures": dict(self.temperatures)}

    def temperature(self, kind: str) -> float:
        return self.temperatures.get(kind, 1.0)


def apply_calibration(output: GroupOutput, kind: str, calibration: Calibration | None) -> GroupOutput:
    if calibration is None:
        return output
    temperature = calibration.temperature(kind)
    if abs(temperature - 1.0) < 1e-9:
        return output
    logits = output.logits / temperature
    probabilities = torch.softmax(logits, dim=-1)
    expected = None
    if kind == SCORE:
        indices = torch.arange(len(logits), device=logits.device, dtype=probabilities.dtype)
        expected = torch.sum(indices * probabilities)
    return GroupOutput(probabilities=probabilities, logits=logits, expected=expected)


def _nll(logits: torch.Tensor, target: torch.Tensor, temperature: float) -> float:
    scaled = logits / temperature
    log_probs = torch.log_softmax(scaled, dim=-1)
    return float(-(target * log_probs).sum().item())


def _teacher_target(request: Request, question) -> torch.Tensor | None:
    probabilities = teacher_distribution(question, request.teacher.get(question.qid))
    if probabilities is None:
        return None
    return torch.tensor(probabilities, dtype=torch.float32)


def _target_for(request: Request, question, target_kind: str, path_count: int) -> torch.Tensor | None:
    builders = {
        "gold": lambda: gold_target(question, request.gold.get(question.qid), path_count),
        "teacher": lambda: _teacher_target(request, question),
    }
    for kind in ([target_kind] + [other for other in builders if other != target_kind]):
        target = builders[kind]()
        if target is not None:
            return target
    return None


def collect_logits(
    model, requests: list[Request], mode: str = "fresh", target_kind: str = "gold", device: str = "auto"
) -> dict:
    """Group raw logits and targets by primitive for calibration fitting."""
    model.eval_mode(device)
    buckets: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {CHOICE: [], BOOLEAN: [], SCORE: []}
    with torch.no_grad():
        encoded = model.encode(requests)
        outputs = model.run_encoded(encoded, mode)
        for group, output in zip(encoded.groups, outputs):
            request = encoded.requests[group.request_index]
            question = request.questions[group.question_index]
            target = _target_for(request, question, target_kind, len(group.path_indices))
            if target is None:
                continue
            buckets[group.kind].append((output.logits.detach().cpu().float(), target.cpu().float()))
    return buckets


def _fit_one(pairs: list[tuple[torch.Tensor, torch.Tensor]]) -> float:
    if not pairs:
        return 1.0
    low, high, steps = math.log(0.1), math.log(4.0), 64
    grid = [math.exp(low + index * (high - low) / (steps - 1)) for index in range(steps)]
    grid.append(1.0)
    best_temperature = 1.0
    best_loss = float("inf")
    for temperature in grid:
        loss = sum(_nll(logits, target, temperature) for logits, target in pairs)
        if loss < best_loss:
            best_loss = loss
            best_temperature = temperature
    return round(best_temperature, 4)


def fit_calibration(
    model, requests: list[Request], mode: str = "fresh", target: str = "gold", device: str = "auto"
) -> tuple[Calibration, dict]:
    if target not in {"gold", "teacher"}:
        raise ValueError("calibration target must be 'gold' or 'teacher'")
    buckets = collect_logits(model, requests, mode=mode, target_kind=target, device=device)
    temperatures = {kind: _fit_one(pairs) for kind, pairs in buckets.items()}
    before = {}
    after = {}
    for kind, pairs in buckets.items():
        if not pairs:
            continue
        temperature = temperatures[kind]
        before[kind] = sum(_nll(logits, target_vector, 1.0) for logits, target_vector in pairs) / len(pairs)
        after[kind] = sum(_nll(logits, target_vector, temperature) for logits, target_vector in pairs) / len(pairs)
    report = {
        "target": target,
        "questions": {kind: len(pairs) for kind, pairs in buckets.items()},
        "temperature": temperatures,
        "nll_before": {kind: round(value, 6) for kind, value in before.items()},
        "nll_after": {kind: round(value, 6) for kind, value in after.items()},
    }
    return Calibration(temperatures=temperatures), report
