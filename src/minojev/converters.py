"""Convert permissive public datasets into minojev decision requests.

The upstream datasets provide supervision; this layer builds the Jev-shaped
request (state, question, candidates, teacher) deterministically. Every request
carries its source name as ``family`` so metrics can be split by origin, and
conversion is pure Python so it is testable without network access.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from .types import Request, make_boolean_question, make_choice_question, make_score_question

LEVELS = ("none", "low", "medium", "high", "severe")


@dataclass(frozen=True)
class SourceSpec:
    name: str
    license: str
    primitive: str
    group: str
    labels: tuple[str, ...] = ()
    text_field: str = "text"
    label_field: str = "label"
    label_text_field: str | None = None
    config: str | None = None
    split: str = "train"
    notes: str = ""
    local: bool = False


def resolve_answer(row: dict, spec: SourceSpec) -> str:
    """Read the answer whether the label is an index, a string, or a text column."""
    if spec.label_text_field and row.get(spec.label_text_field) is not None:
        return str(row[spec.label_text_field])
    value = row[spec.label_field]
    if isinstance(value, str) and not value.lstrip("-").isdigit():
        return value
    return spec.labels[int(value)]


def sample_candidates(rng: random.Random, labels: list[str], answer: str, k: int) -> list[str]:
    """Answer plus random negatives, preserving the full candidate ordering."""
    others = [label for label in labels if label != answer]
    rng.shuffle(others)
    negatives = others[: max(0, min(k - 1, len(others)))]
    candidates = [answer] + negatives
    rng.shuffle(candidates)
    return candidates


def peaked(candidates: list[str], answer: str, peak: float = 0.85) -> dict[str, float]:
    other = (1 - peak) / max(len(candidates) - 1, 1)
    return {candidate: (peak if candidate == answer else other) for candidate in candidates}


def choice_question(qid: str, instructions: str, candidates: list[str]) -> Any:
    return make_choice_question(qid, instructions, {candidate: candidate for candidate in candidates})


def classification_to_choice(
    row: dict,
    spec: SourceSpec,
    rng: random.Random,
    max_candidates: int = 8,
) -> Request:
    answer = resolve_answer(row, spec)
    labels = list(spec.labels)
    if len(labels) <= max_candidates:
        candidates = labels
    else:
        k = rng.randint(2, max_candidates)
        candidates = sample_candidates(rng, labels, answer, k)
    question = choice_question("category", f"Which category applies to this text?", candidates)
    question.family = spec.name
    return Request(
        id=f"{spec.name}-{row.get('__index__', 0)}",
        state={spec.text_field: row[spec.text_field]},
        questions=[question],
        gold={"category": answer},
        teacher={"category": peaked(candidates, answer)},
    )


def binary_to_boolean(row: dict, spec: SourceSpec, rng: random.Random) -> Request:
    positive_label = spec.labels[1]
    negative_label = spec.labels[0]
    answer = resolve_answer(row, spec)
    ask_about = positive_label if rng.random() < 0.5 else negative_label
    truth = answer == ask_about
    question = make_boolean_question(
        "claim",
        f"Is this text {ask_about}?",
        {"true": f"the text is {ask_about}", "false": f"the text is not {ask_about}"},
    )
    question.family = spec.name
    return Request(
        id=f"{spec.name}-{row.get('__index__', 0)}",
        state={spec.text_field: row[spec.text_field]},
        questions=[question],
        gold={"claim": truth},
        teacher={"claim": {"true": 0.9 if truth else 0.1, "false": 0.1 if truth else 0.9}},
    )


def _fraction_to_levels(fraction: float) -> list[float]:
    """A soft distribution over ordered levels centered on the observed fraction."""
    center = max(0.0, min(1.0, fraction)) * (len(LEVELS) - 1)
    weights = []
    for index in range(len(LEVELS)):
        weights.append(2.718281828 ** (-1.6 * abs(index - center)))
    total = sum(weights)
    return [weight / total for weight in weights]


def continuous_to_score(
    row: dict,
    spec: SourceSpec,
    rng: random.Random,
    value_field: str,
) -> Request:
    fraction = float(row[value_field])
    distribution = _fraction_to_levels(fraction)
    answer = max(range(len(distribution)), key=distribution.__getitem__)
    question = make_score_question("intensity", "How strong is the signal in this text?", list(LEVELS))
    question.family = spec.name
    return Request(
        id=f"{spec.name}-{row.get('__index__', 0)}",
        state={spec.text_field: row[spec.text_field]},
        questions=[question],
        gold={"intensity": answer},
        teacher={"intensity": {str(index): round(value, 6) for index, value in enumerate(distribution)}},
    )


def continuous_to_boolean(
    row: dict,
    spec: SourceSpec,
    rng: random.Random,
    value_field: str,
    threshold: float,
) -> Request:
    fraction = float(row[value_field])
    truth = fraction >= threshold
    question = make_boolean_question(
        "gate",
        f"Should this text be blocked by the policy threshold?",
        {"true": "the signal is at or above the threshold", "false": "the signal is below the threshold"},
    )
    question.family = spec.name
    # Soft target reflects distance from the threshold, capped at 0.95.
    confidence = min(0.5 + abs(fraction - threshold) * 3.0, 0.95)
    return Request(
        id=f"{spec.name}-{row.get('__index__', 0)}",
        state={spec.text_field: row[spec.text_field]},
        questions=[question],
        gold={"gate": truth},
        teacher={"gate": {"true": confidence if truth else 1 - confidence, "false": 1 - confidence if truth else confidence}},
    )


def nli_to_choice(row: dict, spec: SourceSpec, rng: random.Random) -> Request:
    order = spec.labels
    answer = resolve_answer(row, spec)
    question = choice_question("relation", "What is the relationship between the premise and the hypothesis?", list(order))
    question.family = spec.name
    return Request(
        id=f"{spec.name}-{row.get('__index__', 0)}",
        state={"premise": row["premise"], "hypothesis": row["hypothesis"]},
        questions=[question],
        gold={"relation": answer},
        teacher={"relation": peaked(list(order), answer)},
    )


def nli_to_boolean(row: dict, spec: SourceSpec, rng: random.Random) -> Request:
    answer = spec.labels[int(row[spec.label_field])]
    truth = answer == "entailment"
    question = make_boolean_question(
        "entails",
        "Does the premise entail the hypothesis?",
        {"true": "the premise entails the hypothesis", "false": "the premise does not entail the hypothesis"},
    )
    question.family = spec.name
    return Request(
        id=f"{spec.name}-{row.get('__index__', 0)}",
        state={"premise": row["premise"], "hypothesis": row["hypothesis"]},
        questions=[question],
        gold={"entails": truth},
        teacher={"entails": {"true": 0.9 if truth else 0.1, "false": 0.1 if truth else 0.9}},
    )


CONVERTERS: dict[str, Callable[..., Request]] = {
    "choice": classification_to_choice,
    "boolean": binary_to_boolean,
    "score": continuous_to_score,
    "gate": continuous_to_boolean,
    "nli-choice": nli_to_choice,
    "nli-boolean": nli_to_boolean,
}


def convert_rows(rows: Iterable[dict], spec: SourceSpec, seed: int = 17, **options) -> list[Request]:
    rng = random.Random(seed)
    converter = CONVERTERS[spec.primitive]
    requests = []
    for index, row in enumerate(rows):
        row = dict(row)
        row["__index__"] = index
        requests.append(converter(row, spec, rng, **options))
    return requests
