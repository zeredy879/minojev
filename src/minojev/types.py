"""Typed request validation and normalization for minojev decisions.

A request carries one state and any number of independent questions. Each
question is one of three primitives:

- ``choice``: 2-255 described candidates; output is a distribution over them.
- ``boolean``: one proposition with optional false/true criteria; output is
  ``P(true)``.
- ``score``: 2-10 ordered levels; output is a level distribution and its
  probability-weighted expected level.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

CHOICE = "choice"
BOOLEAN = "boolean"
SCORE = "score"
KINDS = (CHOICE, BOOLEAN, SCORE)

MIN_CHOICE = 2
MAX_CHOICE = 255
MIN_LEVELS = 2
MAX_LEVELS = 10

BOOLEAN_CANDIDATE_IDS = ("false", "true")


class ValidationError(ValueError):
    """Raised when a request violates the decision contract."""


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    description: str


@dataclass
class Question:
    qid: str
    kind: str
    instructions: str
    candidates: list[Candidate]
    criteria: dict[str, str] = field(default_factory=dict)
    family: str = ""

    @property
    def candidate_ids(self) -> list[str]:
        return [candidate.candidate_id for candidate in self.candidates]


@dataclass
class Request:
    id: str
    state: Any
    questions: list[Question]
    gold: dict[str, Any] = field(default_factory=dict)
    teacher: dict[str, dict[str, float]] = field(default_factory=dict)

    @property
    def state_key(self) -> str:
        return serialize_state(self.state)

    def question(self, qid: str) -> Question:
        for question in self.questions:
            if question.qid == qid:
                return question
        raise KeyError(qid)


def serialize_state(state: Any) -> str:
    """Canonical text serialization used for the model input."""
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _validate_state(state: Any) -> None:
    if isinstance(state, str):
        if not state.strip():
            raise ValidationError("state must be nonempty")
        return
    if isinstance(state, (dict, list)):
        if not state:
            raise ValidationError("state must be nonempty")
        try:
            json.dumps(state, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValidationError("state must be finite JSON-compatible data") from error
        return
    raise ValidationError("state must be a nonempty string, object, or array")


def _validate_description(text: Any, label: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValidationError(f"{label} must be a nonempty string")
    return text


def make_choice_question(
    qid: str,
    instructions: str,
    options: dict[str, str] | list[Candidate] | list[tuple[str, str]],
) -> Question:
    instruction_text = _validate_description(instructions, "choice instructions")
    if isinstance(options, dict):
        pairs = list(options.items())
    else:
        pairs = [(item.candidate_id, item.description) if isinstance(item, Candidate) else tuple(item) for item in options]
    if not MIN_CHOICE <= len(pairs) <= MAX_CHOICE:
        raise ValidationError(f"choice needs {MIN_CHOICE}-{MAX_CHOICE} candidates, got {len(pairs)}")
    candidates = []
    for candidate_id, description in pairs:
        candidate_id = _validate_description(candidate_id, "candidate id")
        description = _validate_description(description, "candidate description")
        candidates.append(Candidate(candidate_id, description))
    if len({candidate.candidate_id for candidate in candidates}) != len(candidates):
        raise ValidationError("candidate ids must be unique")
    return Question(qid=qid, kind=CHOICE, instructions=instruction_text, candidates=candidates)


def make_boolean_question(
    qid: str,
    instructions: str,
    criteria: dict[str, str] | None = None,
) -> Question:
    instruction_text = _validate_description(instructions, "boolean instructions")
    normalized: dict[str, str] = {}
    if criteria:
        unknown = set(criteria) - set(BOOLEAN_CANDIDATE_IDS)
        if unknown:
            raise ValidationError(f"boolean criteria keys must be false/true, got {sorted(unknown)}")
        for key, value in criteria.items():
            normalized[key] = _validate_description(value, f"boolean {key} criterion")
    defaults = {
        "false": "The proposition is false.",
        "true": "The proposition is true.",
    }
    candidates = [Candidate(key, normalized.get(key, defaults[key])) for key in BOOLEAN_CANDIDATE_IDS]
    return Question(qid=qid, kind=BOOLEAN, instructions=instruction_text, candidates=candidates, criteria=normalized)


def make_score_question(qid: str, instructions: str, levels: list[str]) -> Question:
    instruction_text = _validate_description(instructions, "score instructions")
    if not isinstance(levels, list) or not MIN_LEVELS <= len(levels) <= MAX_LEVELS:
        raise ValidationError(f"score needs {MIN_LEVELS}-{MAX_LEVELS} ordered levels")
    candidates = []
    for index, level in enumerate(levels):
        candidates.append(Candidate(str(index), _validate_description(level, "score level")))
    return Question(qid=qid, kind=SCORE, instructions=instruction_text, candidates=candidates)


def question_from_object(qid: str, obj: dict) -> Question:
    if not isinstance(obj, dict):
        raise ValidationError(f"question {qid} must be an object")
    instructions = obj.get("instructions", obj.get("question"))
    kind = str(obj.get("type", CHOICE)).lower()
    if kind in {"noul", "bool"}:
        kind = BOOLEAN
    family = str(obj.get("family", ""))
    if kind == CHOICE:
        options = obj.get("options")
        if options is None and isinstance(obj.get("criteria"), dict):
            options = obj["criteria"]
        if isinstance(options, dict):
            question = make_choice_question(qid, instructions, options)
        elif isinstance(options, list):
            pairs = []
            for index, option in enumerate(options):
                if isinstance(option, dict):
                    pairs.append((str(option.get("id", option.get("key", index))), option.get("description", option.get("text"))))
                else:
                    pairs.append((str(index), str(option)))
            question = make_choice_question(qid, instructions, pairs)
        else:
            raise ValidationError(f"choice question {qid} needs an options list or map")
    elif kind == BOOLEAN:
        question = make_boolean_question(qid, instructions, obj.get("criteria"))
    elif kind == SCORE:
        question = make_score_question(qid, instructions, obj.get("levels"))
    else:
        raise ValidationError(f"unknown question type {kind!r} in {qid!r}")
    question.family = family
    return question


def request_from_object(obj: dict, index: int = 0) -> Request:
    if not isinstance(obj, dict):
        raise ValidationError("request must be a JSON object")
    request_id = obj.get("id", f"row-{index}")
    request_id = _validate_description(request_id, "request id")
    _validate_state(obj.get("state"))
    if "questions" in obj:
        raw_questions = obj["questions"]
        if not isinstance(raw_questions, dict) or not raw_questions:
            raise ValidationError("questions must be a nonempty map")
        questions = [question_from_object(qid, raw) for qid, raw in raw_questions.items()]
    elif "question" in obj:
        flat = dict(obj)
        flat.setdefault("type", CHOICE)
        questions = [question_from_object("q0", flat)]
    else:
        raise ValidationError("request needs a questions map or a flat question")
    if len({question.qid for question in questions}) != len(questions):
        raise ValidationError("question ids must be unique within a request")
    gold = obj.get("gold", {}) or {}
    teacher = obj.get("teacher", {}) or {}
    if not isinstance(gold, dict) or not isinstance(teacher, dict):
        raise ValidationError("gold and teacher must be maps")
    return Request(id=request_id, state=obj["state"], questions=questions, gold=gold, teacher=teacher)


def request_to_object(request: Request) -> dict:
    questions: dict[str, Any] = {}
    for question in request.questions:
        if question.kind == CHOICE:
            entry = {
                "type": CHOICE,
                "instructions": question.instructions,
                "options": {candidate.candidate_id: candidate.description for candidate in question.candidates},
            }
        elif question.kind == BOOLEAN:
            entry = {"type": BOOLEAN, "instructions": question.instructions}
            if question.criteria:
                entry["criteria"] = dict(question.criteria)
        else:
            entry = {
                "type": SCORE,
                "instructions": question.instructions,
                "levels": [candidate.description for candidate in question.candidates],
            }
        if question.family:
            entry["family"] = question.family
        questions[question.qid] = entry
    obj: dict[str, Any] = {"id": request.id, "state": request.state, "questions": questions}
    if request.gold:
        obj["gold"] = dict(request.gold)
    if request.teacher:
        obj["teacher"] = dict(request.teacher)
    return obj


def teacher_distribution(question: Question, target: dict[str, float] | None) -> list[float] | None:
    """Normalize a declared teacher distribution over candidate ids, if complete."""
    if not target:
        return None
    ids = question.candidate_ids
    if set(target) != set(ids):
        raise ValidationError(f"teacher keys must match candidates for {question.qid}: {sorted(target)} != {sorted(ids)}")
    values = [float(target[candidate_id]) for candidate_id in ids]
    total = sum(values)
    if any(value < 0 or value != value for value in values) or total <= 0:
        raise ValidationError(f"teacher distribution for {question.qid} must be nonnegative and sum above zero")
    return [value / total for value in values]
