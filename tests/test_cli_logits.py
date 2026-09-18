import json
import os
import subprocess
import sys
from pathlib import Path

from minojev import ByteTokenizer
from minojev.logits import build_prompt, letter_slot_ids, score_logits
from minojev.types import Request, make_boolean_question, make_choice_question, make_score_question


def test_letter_slots_round_trip():
    tokenizer = ByteTokenizer()
    slots = letter_slot_ids(tokenizer)
    assert len(slots) == 16
    assert tokenizer.decode([slots[0]]) == "A"
    assert tokenizer.decode([slots[15]]) == "P"


def test_logits_prompt_contains_options():
    request = Request(
        id="r",
        state={"k": "v"},
        questions=[make_score_question("q", "rate it", ["low", "high"])],
    )
    prompt = build_prompt(request, request.questions[0])
    assert "A: low" in prompt
    assert "B: high" in prompt
    assert prompt.endswith("Answer:")


def test_logits_readout_probabilities(tiny_model):
    request = Request(
        id="r",
        state="s",
        questions=[
            make_choice_question("c", "pick", {"a": "first", "b": "second", "c": "third"}),
            make_boolean_question("b", "claim?"),
        ],
    )
    records = score_logits(tiny_model.backbone, tiny_model.tokenizer, [request])
    assert len(records) == 2
    for record in records:
        assert abs(sum(record["probabilities"]) - 1.0) < 1e-5
        assert record["decode_steps"] == 0
        assert record["mode"] == "logits"
    assert len(records[0]["probabilities"]) == 3
    assert len(records[1]["probabilities"]) == 2


def test_cli_end_to_end(tmp_path):
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    train_path = tmp_path / "train.jsonl"
    dev_path = tmp_path / "dev.jsonl"
    checkpoint = tmp_path / "run" / "checkpoint"
    scores = tmp_path / "scores.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    demo = tmp_path / "demo.json"

    def run(*args):
        result = subprocess.run(
            [sys.executable, "-m", "minojev.cli", *args],
            capture_output=True,
            text=True,
            env=env,
            timeout=600,
        )
        assert result.returncode == 0, result.stderr
        return result.stdout

    run("synth", "--out", str(train_path), "--count", "6", "--seed", "5", "--split", "train")
    run("synth", "--out", str(dev_path), "--count", "3", "--seed", "5", "--split", "dev")
    summary = json.loads(
        run(
            "train",
            "--train",
            str(train_path),
            "--dev",
            str(dev_path),
            "--output-dir",
            str(tmp_path / "run"),
            "--steps",
            "12",
            "--head-steps",
            "4",
            "--eval-every",
            "8",
            "--hidden-size",
            "32",
            "--layers",
            "1",
            "--heads",
            "2",
            "--intermediate-size",
            "64",
            "--device",
            "cpu",
        )
    )
    assert summary["best_step"] is not None
    expected = sum(len(json.loads(line)["questions"]) for line in dev_path.read_text().splitlines() if line.strip())
    output = json.loads(run("score", "--checkpoint", str(checkpoint), "--input", str(dev_path), "--output", str(scores), "--device", "cpu"))
    assert output["decode_steps"] == 0
    assert len(scores.read_text().splitlines()) == expected
    report = json.loads(
        run("evaluate", "--checkpoint", str(checkpoint), "--input", str(dev_path), "--predictions", str(predictions), "--device", "cpu")
    )
    assert report["questions"] == expected
    bundle = json.loads(
        run("demo", "--checkpoint", str(checkpoint), "--output", str(demo), "--count", "2", "--seed", "5", "--device", "cpu")
    )
    assert bundle["records"] == len(json.loads(demo.read_text())["records"])
    info = json.loads(run("info", "--checkpoint", str(checkpoint)))
    assert info["format"] == "minojev-checkpoint"
