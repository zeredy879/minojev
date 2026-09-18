from minojev import BOOLEAN, CHOICE, SCORE
from minojev.metrics import aggregate
from minojev.synth import generate_request, generate_split
from minojev.types import teacher_distribution


def test_generation_is_deterministic():
    first = generate_request(3, seed=5)
    second = generate_request(3, seed=5)
    assert first.state == second.state
    assert [question.instructions for question in first.questions] == [
        question.instructions for question in second.questions
    ]
    assert generate_request(3, seed=6).state != first.state or generate_request(4, seed=5).state != first.state


def test_splits_are_disjoint():
    train = {request.id for request in generate_split(8, seed=1, split="train")}
    test = {request.id for request in generate_split(8, seed=1, split="test")}
    assert not train & test


def test_teacher_distributions_are_valid():
    for request in generate_split(12, seed=3, split="dev"):
        for question in request.questions:
            probabilities = teacher_distribution(question, request.teacher[question.qid])
            assert probabilities is not None
            assert abs(sum(probabilities) - 1.0) < 1e-9
            assert all(0 <= value <= 1 for value in probabilities)
            if question.kind == SCORE:
                assert probabilities.index(max(probabilities)) == int(request.gold[question.qid])
            if question.kind == CHOICE:
                assert question.candidate_ids[probabilities.index(max(probabilities))] == request.gold[question.qid]
            if question.kind == BOOLEAN:
                predicted = probabilities[1] > probabilities[0]
                assert predicted == request.gold[question.qid]


def test_aggregate_metrics():
    records = [
        {
            "type": CHOICE,
            "candidate_ids": ["a", "b"],
            "probabilities": [0.8, 0.2],
            "gold": "a",
            "correct": True,
            "teacher": {"a": 0.75, "b": 0.25},
            "decode_steps": 0,
        },
        {
            "type": BOOLEAN,
            "candidate_ids": ["false", "true"],
            "probabilities": [0.6, 0.4],
            "gold": True,
            "correct": False,
            "decode_steps": 0,
        },
    ]
    summary = aggregate(records)
    assert summary["questions"] == 2
    assert summary["accuracy"] == 0.5
    assert summary["decode_steps"] == 0
    assert abs(summary["distribution_error"] - ((0.8 - 0.75) ** 2 + (0.2 - 0.25) ** 2)) < 1e-9
