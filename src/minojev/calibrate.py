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
    model,
    requests: list[Request],
    mode: str = "fresh",
    target_kind: str = "gold",
    device: str = "auto",
    batch_requests: int = 16,
    max_memory_gb: float | None = None,
) -> dict:
    """Group raw logits and targets by primitive for calibration fitting.

    Requests are forwarded in chunks so peak activation memory stays bounded
    (a single forward over a full dev split previously exhausted unified
    memory on Apple Silicon).
    """
    from .monitor import MemoryBudget, release_memory

    model.eval_mode(device)
    budget = MemoryBudget(max_gb=max_memory_gb)
    buckets: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {CHOICE: [], BOOLEAN: [], SCORE: []}
    chunk = max(1, int(batch_requests))
    with torch.no_grad():
        for start in range(0, len(requests), chunk):
            subset = requests[start : start + chunk]
            budget.check(f"calibration chunk {start}")
            encoded = model.encode(subset)
            outputs = model.run_encoded(encoded, mode)
            for group, output in zip(encoded.groups, outputs):
                request = encoded.requests[group.request_index]
                question = request.questions[group.question_index]
                target = _target_for(request, question, target_kind, len(group.path_indices))
                if target is None:
                    continue
                buckets[group.kind].append((output.logits.detach().cpu().float(), target.cpu().float()))
            release_memory()
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


def _grid() -> list[float]:
    low, high, steps = math.log(0.1), math.log(4.0), 64
    values = [math.exp(low + index * (high - low) / (steps - 1)) for index in range(steps)]
    values.append(1.0)
    return values


def _nll_from_probabilities(probabilities: list[float], target: list[float], temperature: float) -> float:
    """Temperature scaling of softmax probabilities: p^(1/T) renormalized."""
    scaled = [max(probability, 1e-12) ** (1.0 / temperature) for probability in probabilities]
    total = sum(scaled)
    normalized = [value / total for value in scaled]
    return -sum(t * math.log(max(p, 1e-12)) for t, p in zip(target, normalized))


def _record_target(record: dict, target: str) -> list[float] | None:
    ids = record["candidate_ids"]
    if target == "teacher" and record.get("teacher"):
        values = [float(record["teacher"].get(candidate_id, 0.0)) for candidate_id in ids]
        total = sum(values)
        return [value / total for value in values] if total > 0 else None
    gold = record.get("gold")
    if gold is None:
        return None
    if record["type"] == BOOLEAN:
        index = 1 if gold is True or gold in (1, "true", "True") else 0
    elif record["type"] == "score":
        index = int(gold)
    else:
        index = ids.index(gold)
    values = [0.0] * len(ids)
    values[index] = 1.0
    return values


def fit_calibration_from_records(records: list[dict], target: str = "gold") -> tuple[Calibration, dict]:
    """Fit temperatures from already-scored records (no extra forward pass)."""
    if target not in {"gold", "teacher"}:
        raise ValueError("calibration target must be 'gold' or 'teacher'")
    buckets: dict[str, list[tuple[list[float], list[float]]]] = {CHOICE: [], BOOLEAN: [], SCORE: []}
    for record in records:
        if record.get("type") not in buckets:
            continue
        target_values = _record_target(record, target)
        if target_values is None:
            continue
        buckets[record["type"]].append((record["probabilities"], target_values))
    temperatures: dict[str, float] = {}
    before: dict[str, float] = {}
    after: dict[str, float] = {}
    grid = _grid()
    for kind, pairs in buckets.items():
        if not pairs:
            temperatures[kind] = 1.0
            continue
        best_temperature, best_loss = 1.0, float("inf")
        for temperature in grid:
            loss = sum(_nll_from_probabilities(probabilities, target, temperature) for probabilities, target in pairs)
            if loss < best_loss:
                best_loss, best_temperature = loss, temperature
        temperatures[kind] = round(best_temperature, 4)
        before[kind] = sum(_nll_from_probabilities(p, t, 1.0) for p, t in pairs) / len(pairs)
        after[kind] = sum(_nll_from_probabilities(p, t, best_temperature) for p, t in pairs) / len(pairs)
    report = {
        "target": target,
        "method": "record-based temperature scaling",
        "questions": {kind: len(pairs) for kind, pairs in buckets.items()},
        "temperature": temperatures,
        "nll_before": {kind: round(value, 6) for kind, value in before.items()},
        "nll_after": {kind: round(value, 6) for kind, value in after.items()},
    }
    return Calibration(temperatures=temperatures), report


def fit_calibration(
    model,
    requests: list[Request],
    mode: str = "fresh",
    target: str = "gold",
    device: str = "auto",
    batch_requests: int = 16,
    max_memory_gb: float | None = None,
) -> tuple[Calibration, dict]:
    if target not in {"gold", "teacher"}:
        raise ValueError("calibration target must be 'gold' or 'teacher'")
    buckets = collect_logits(
        model,
        requests,
        mode=mode,
        target_kind=target,
        device=device,
        batch_requests=batch_requests,
        max_memory_gb=max_memory_gb,
    )
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
