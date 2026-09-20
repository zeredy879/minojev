import json
import random
from types import SimpleNamespace

import pytest

from minojev.converters import (
    SourceSpec,
    binary_to_boolean,
    classification_to_choice,
    convert_rows,
    continuous_to_boolean,
    continuous_to_score,
    nli_to_choice,
)
from minojev.types import teacher_distribution


def _choice_spec(labels=("a", "b", "c")):
    return SourceSpec(name="unit-choice", license="mit", primitive="choice", group="train", labels=labels)


def test_classification_to_choice_contract():
    spec = _choice_spec()
    rows = [{"text": f"example {index}", "label": index % 3} for index in range(9)]
    requests = convert_rows(rows, spec, seed=1)
    assert len(requests) == 9
    for request in requests:
        question = request.questions[0]
        assert question.kind == "choice"
        answer = request.gold["category"]
        assert answer in question.candidate_ids
        probabilities = teacher_distribution(question, request.teacher["category"])
        assert abs(sum(probabilities) - 1.0) < 1e-9
        assert question.candidate_ids[probabilities.index(max(probabilities))] == answer


def test_high_cardinality_choice_samples_candidates():
    labels = tuple(f"intent-{index}" for index in range(77))
    spec = _choice_spec(labels)
    rows = [{"text": f"t{index}", "label": index % 77} for index in range(20)]
    requests = convert_rows(rows, spec, seed=3, max_candidates=8)
    sizes = [len(request.questions[0].candidate_ids) for request in requests]
    assert all(2 <= size <= 8 for size in sizes)
    assert any(size > 2 for size in sizes)


def test_binary_to_boolean_truth_logic():
    spec = SourceSpec(name="unit-bool", license="mit", primitive="boolean", group="train", labels=("negative", "positive"))
    rng = random.Random(0)
    rows = [{"text": "good", "label": 1}, {"text": "bad", "label": 0}]
    for row in rows:
        request = binary_to_boolean(row, spec, rng)
        question = request.questions[0]
        probabilities = teacher_distribution(question, request.teacher["claim"])
        predicted = probabilities[1] > probabilities[0]
        assert predicted == request.gold["claim"]


def test_continuous_to_score_is_soft_and_ordered():
    spec = SourceSpec(name="unit-score", license="mit", primitive="score", group="train")
    row = {"text": "mild", "toxicity": 0.15}
    request = continuous_to_score(row, spec, random.Random(0), value_field="toxicity")
    question = request.questions[0]
    probabilities = teacher_distribution(question, request.teacher["intensity"])
    assert len(probabilities) == 5
    assert probabilities[0] > probabilities[4]
    assert request.gold["intensity"] == probabilities.index(max(probabilities))
    high = continuous_to_score({"text": "bad", "toxicity": 0.95}, spec, random.Random(0), value_field="toxicity")
    high_probabilities = teacher_distribution(high.questions[0], high.teacher["intensity"])
    assert high_probabilities[4] > high_probabilities[0]


def test_continuous_to_boolean_soft_targets():
    spec = SourceSpec(name="unit-gate", license="mit", primitive="gate", group="train")
    below = continuous_to_boolean({"text": "a", "score": 0.05}, spec, random.Random(0), value_field="score", threshold=0.5)
    above = continuous_to_boolean({"text": "b", "score": 0.95}, spec, random.Random(0), value_field="score", threshold=0.5)
    assert below.gold["gate"] is False and above.gold["gate"] is True
    assert above.teacher["gate"]["true"] > 0.9
    assert below.teacher["gate"]["true"] < 0.1


def test_nli_to_choice_three_way():
    spec = SourceSpec(name="unit-nli", license="mit", primitive="nli-choice", group="train", labels=("entailment", "neutral", "contradiction"))
    row = {"premise": "A cat sits on a mat.", "hypothesis": "There is a cat.", "label": 0}
    request = nli_to_choice(row, spec, random.Random(0))
    assert request.gold["relation"] == "entailment"
    assert request.questions[0].candidate_ids == ["entailment", "neutral", "contradiction"]
    assert request.state["premise"].startswith("A cat")


def test_build_dataset_source_isolation(tmp_path, monkeypatch):
    import minojev.dataset_build as builder

    registry = {
        "train-choice": SourceSpec(name="unit/train-choice", license="mit", primitive="choice", group="train"),
        "train-bool": SourceSpec(name="unit/train-bool", license="mit", primitive="boolean", group="train"),
        "ood-choice": SourceSpec(name="unit/ood-choice", license="mit", primitive="choice", group="ood"),
    }
    monkeypatch.setattr(builder, "SOURCES", registry)

    def fake_load_rows(spec, limit):
        if spec.name == "unit/train-bool":
            rows = [{"text": f"t{i}", "label": i % 2} for i in range(10)]
            features = {"label": SimpleNamespace(names=("negative", "positive"))}
        else:
            rows = [{"text": f"t{i}", "label": i % 3} for i in range(10)]
            features = {"label": SimpleNamespace(names=("a", "b", "c"))}
        return rows, features

    monkeypatch.setattr(builder, "load_rows", fake_load_rows)
    result = builder.build_dataset(builder.BuildConfig(out_dir=str(tmp_path), per_source=10, per_ood=10, seed=5))
    assert result["splits"]["train"] + result["splits"]["dev"] + result["splits"]["test"] == 20
    assert result["splits"]["ood"] == 10
    sources = {entry["key"] for entry in json.loads((tmp_path / "general-dataset-card.json").read_text())["sources"]}
    assert sources == {"train-choice", "train-bool", "ood-choice"}
    ood_rows = [json.loads(line) for line in (tmp_path / "general-ood.jsonl").read_text().splitlines()]
    assert all(row["source"] == "unit/ood-choice" for row in ood_rows)
    train_rows = [json.loads(line) for line in (tmp_path / "general-train.jsonl").read_text().splitlines()]
    assert all(row["source"].startswith("unit/train") for row in train_rows)
    assert all("license" in row for row in train_rows)
