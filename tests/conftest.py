import pytest

from minojev import (
    BOOLEAN,
    CHOICE,
    SCORE,
    DecisionModel,
    Request,
    make_boolean_question,
    make_choice_question,
    make_score_question,
    request_from_object,
    request_to_object,
)


@pytest.fixture
def tiny_model():
    return DecisionModel.new(hidden_size=32, num_layers=1, num_heads=2, intermediate_size=64)


@pytest.fixture
def sample_requests():
    first = Request(
        id="r1",
        state={"color": "blue", "shape": "round", "size": "large"},
        questions=[
            make_choice_question("q0", "Which value belongs to 'shape'?", {"round": "round", "red": "red", "tiny": "tiny"}),
            make_boolean_question(
                "q1",
                "Does the state support that size is large?",
                {"true": "size is large", "false": "size is not large"},
            ),
            make_score_question("q2", "How large is the size?", ["tiny", "small", "large", "huge"]),
        ],
        gold={"q0": "round", "q1": True, "q2": 2},
        teacher={
            "q0": {"round": 0.8, "red": 0.1, "tiny": 0.1},
            "q1": {"true": 0.9, "false": 0.1},
            "q2": {"0": 0.05, "1": 0.15, "2": 0.6, "3": 0.2},
        },
    )
    second = Request(
        id="r2",
        state={"color": "green"},
        questions=[
            make_choice_question("q0", "Which value belongs to 'color'?", {"red": "red", "green": "green"}),
        ],
        gold={"q0": "green"},
    )
    return [first, second]
