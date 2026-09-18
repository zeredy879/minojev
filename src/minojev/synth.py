"""Deterministic synthetic decision families used for end-to-end training.

Each request describes an attribute table; questions ask for lookups,
comparisons, and support judgments over that table. Teacher distributions are
computed by construction, so training targets are exact and auditable. Splits
are generated from disjoint seed streams, which keeps test requests out of the
training distribution.
"""

from __future__ import annotations

import random

from .types import (
    CHOICE,
    Request,
    make_boolean_question,
    make_choice_question,
    make_score_question,
)

ATTRIBUTES = ("color", "shape", "size", "mood")
VALUES = {
    "color": ["red", "blue", "green"],
    "shape": ["round", "square", "star"],
    "size": ["tiny", "large", "huge"],
    "mood": ["calm", "eager", "tired"],
}
ALL_VALUES = [value for values in VALUES.values() for value in values]
SUPPORT_MATCH = [0.01, 0.03, 0.08, 0.28, 0.60]
SUPPORT_MISS = [0.60, 0.28, 0.08, 0.03, 0.01]
PEAK = 0.85


def _make_state(rng: random.Random) -> dict[str, str]:
    keys = rng.sample(ATTRIBUTES, 3)
    return {key: rng.choice(VALUES[key]) for key in keys}


def _related_values(key: str, exclude: set[str]) -> list[str]:
    return [value for value in VALUES[key] if value not in exclude]


def _distractors(rng: random.Random, exclude: set[str], count: int) -> list[str]:
    pool = [value for value in ALL_VALUES if value not in exclude]
    return rng.sample(pool, min(count, len(pool)))


def _peaked_target(candidates: list[tuple[str, str]], answer: str) -> dict[str, float]:
    other = (1 - PEAK) / (len(candidates) - 1)
    return {candidate_id: (PEAK if candidate_id == answer else other) for candidate_id, _ in candidates}


def _choice_lookup(rng: random.Random, state: dict[str, str], qid: str):
    key = rng.choice(sorted(state))
    answer = state[key]
    candidates = [(value, value) for value in _distractors(rng, {answer}, rng.choice([1, 2, 3]))]
    candidates.append((answer, answer))
    rng.shuffle(candidates)
    question = make_choice_question(qid, f"Which value belongs to the attribute '{key}'?", candidates)
    return question, answer, _peaked_target(candidates, answer)


def _choice_reverse(rng: random.Random, state: dict[str, str], qid: str):
    value = rng.choice(sorted(state.values()))
    answer = next(key for key, current in state.items() if current == value)
    others = [key for key in sorted(state) if key != answer]
    rng.shuffle(others)
    candidates = [(key, key) for key in others]
    candidates.append((answer, answer))
    rng.shuffle(candidates)
    question = make_choice_question(qid, f"Which attribute has the value '{value}'?", candidates)
    return question, answer, _peaked_target(candidates, answer)


def _boolean_match(rng: random.Random, state: dict[str, str], qid: str):
    key = rng.choice(sorted(state))
    matches = rng.random() < 0.5
    if matches:
        value = state[key]
    else:
        value = rng.choice(_related_values(key, {state[key]}))
    question = make_boolean_question(
        qid,
        f"Is {key} equal to {value}?",
        {"true": f"{key} is {value}", "false": f"{key} is not {value}"},
    )
    target = {"true": 0.9 if matches else 0.1, "false": 0.1 if matches else 0.9}
    return question, matches, target


def _boolean_presence(rng: random.Random, state: dict[str, str], qid: str):
    matches = rng.random() < 0.5
    if matches:
        value = rng.choice(sorted(state.values()))
    else:
        value = rng.choice([candidate for candidate in ALL_VALUES if candidate not in set(state.values())])
    question = make_boolean_question(
        qid,
        f"Does the state contain {value}?",
        {"true": f"Some attribute is {value}", "false": f"No attribute is {value}"},
    )
    target = {"true": 0.9 if matches else 0.1, "false": 0.1 if matches else 0.9}
    return question, matches, target


def _score_support(rng: random.Random, state: dict[str, str], qid: str):
    key = rng.choice(sorted(state))
    matches = rng.random() < 0.5
    if matches:
        value = state[key]
    else:
        value = rng.choice(_related_values(key, {state[key]}))
    levels = [
        f"refuted: {key} is not {value}",
        f"unlikely: {key} is probably not {value}",
        f"uncertain: {key} may be {value}",
        f"likely: {key} is probably {value}",
        f"confirmed: {key} is {value}",
    ]
    question = make_score_question(
        qid,
        f"How strongly does the state support the claim that {key} is {value}?",
        levels,
    )
    answer = len(levels) - 1 if matches else 0
    return question, answer, {str(index): probability for index, probability in enumerate(SUPPORT_MATCH if matches else SUPPORT_MISS)}


DEFAULT_FAMILIES = (_choice_lookup, _boolean_match, _score_support)
STRESS_FAMILIES = (_choice_reverse, _boolean_presence)
FAMILIES = DEFAULT_FAMILIES


def _split_offset(split: str) -> int:
    return {"train": 0, "dev": 1_000_000, "test": 2_000_000}.get(split, 3_000_000)


def generate_request(index: int, seed: int = 17, split: str = "train") -> Request:
    rng = random.Random((seed * 1_000_003) ^ (index + _split_offset(split)))
    state = _make_state(rng)
    count = rng.randint(min(2, len(FAMILIES)), len(FAMILIES))
    families = rng.sample(FAMILIES, count)
    questions = []
    gold = {}
    teacher = {}
    for number, family in enumerate(families):
        qid = f"q{number}"
        question, answer, target = family(rng, state, qid)
        question.family = family.__name__.removeprefix("_").replace("_", "-")
        questions.append(question)
        gold[qid] = answer
        teacher[qid] = target
    return Request(id=f"{split}-{index:06d}", state=state, questions=questions, gold=gold, teacher=teacher)


def generate_split(count: int, seed: int = 17, split: str = "train") -> list[Request]:
    return [generate_request(index, seed=seed, split=split) for index in range(count)]
