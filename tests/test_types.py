import pytest

from minojev import (
    BOOLEAN,
    CHOICE,
    MAX_CHOICE,
    SCORE,
    ValidationError,
    make_boolean_question,
    make_choice_question,
    make_score_question,
    request_from_object,
    request_to_object,
)
from minojev.types import teacher_distribution


def test_choice_limits_and_uniqueness():
    too_few = {"only": "one"}
    with pytest.raises(ValidationError):
        make_choice_question("q", "pick", too_few)
    large = {f"c{index}": f"option {index}" for index in range(MAX_CHOICE)}
    question = make_choice_question("q", "pick", large)
    assert len(question.candidates) == MAX_CHOICE
    too_many = {f"c{index}": f"option {index}" for index in range(MAX_CHOICE + 1)}
    with pytest.raises(ValidationError):
        make_choice_question("q", "pick", too_many)
    with pytest.raises(ValidationError):
        make_choice_question("q", "pick", [("a", "same"), ("a", "same")])


def test_score_limits():
    with pytest.raises(ValidationError):
        make_score_question("q", "rate", ["only one"])
    with pytest.raises(ValidationError):
        make_score_question("q", "rate", [str(index) for index in range(11)])


def test_state_validation():
    with pytest.raises(ValidationError):
        request_from_object({"id": "x", "state": "", "question": "q", "options": {"a": "a", "b": "b"}})
    with pytest.raises(ValidationError):
        request_from_object({"id": "x", "state": float("nan"), "question": "q", "options": {"a": "a", "b": "b"}})
    with pytest.raises(ValidationError):
        request_from_object({"id": "x", "state": "ok"})


def test_teacher_distribution_rejects_wrong_keys():
    question = make_choice_question("q", "pick", {"a": "a", "b": "b"})
    with pytest.raises(ValidationError):
        teacher_distribution(question, {"a": 1.0})
    with pytest.raises(ValidationError):
        teacher_distribution(question, {"a": 0.0, "b": 0.0})
    normalized = teacher_distribution(question, {"a": 2.0, "b": 2.0})
    assert normalized == [0.5, 0.5]


def test_flat_and_nested_round_trip():
    flat = {
        "id": "flat",
        "state": "plain text",
        "question": "Which queue?",
        "options": {"a": "Access", "b": "Billing"},
        "gold": {"q0": "a"},
    }
    request = request_from_object(flat)
    assert len(request.questions) == 1
    assert request.questions[0].kind == CHOICE
    assert request.questions[0].candidate_ids == ["a", "b"]
    assert request.gold["q0"] == "a"

    nested = request_to_object(request)
    assert nested["questions"]["q0"]["type"] == CHOICE
    assert request_from_object(nested).state == "plain text"


def test_boolean_defaults_and_score_levels():
    boolean = make_boolean_question("q", "Is the sky blue?")
    assert boolean.kind == BOOLEAN
    assert boolean.candidate_ids == ["false", "true"]
    assert "proposition" in boolean.candidates[1].description
    score = make_score_question("s", "Rate it", ["low", "middle", "high"])
    assert score.kind == SCORE
    assert [candidate.candidate_id for candidate in score.candidates] == ["0", "1", "2"]


def test_question_from_object_supports_aliases():
    choice = request_from_object(
        {
            "id": "x",
            "state": {"a": 1},
            "questions": {"q": {"type": "choice", "question": "pick", "criteria": {"one": "first", "two": "second"}}},
        }
    )
    assert choice.questions[0].candidate_ids == ["one", "two"]
    boolean = request_from_object(
        {"id": "y", "state": "s", "questions": {"q": {"type": "noul", "instructions": "is it?", "criteria": {"true": "yes"}}}}
    )
    assert boolean.questions[0].kind == BOOLEAN
    assert boolean.questions[0].criteria == {"true": "yes"}
