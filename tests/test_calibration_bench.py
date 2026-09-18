import pytest
import torch

from minojev import Calibration, DecisionModel, Request, make_choice_question
from minojev.bench import BenchOptions, benchmark_model, compare_modes
from minojev.calibrate import apply_calibration, fit_calibration
from minojev.logits import score_logits_two_stage
from minojev.metrics import _ece, aggregate
from minojev.synth import generate_split


def _corpus():
    return generate_split(12, seed=21, split="dev")


def test_apply_calibration_preserves_argmax(tiny_model):
    requests = _corpus()
    encoded = tiny_model.encode(requests)
    outputs = tiny_model.run_encoded(encoded, "fresh")
    calibrated = Calibration(temperatures={"choice": 0.3, "boolean": 2.5, "score": 0.5})
    for group, output in zip(encoded.groups, outputs):
        adjusted = apply_calibration(output, group.kind, calibrated)
        assert int(torch.argmax(adjusted.probabilities)) == int(torch.argmax(output.probabilities))
        assert abs(float(adjusted.probabilities.detach().sum()) - 1.0) < 1e-5


def test_fit_calibration_is_monotone_on_dev(tiny_model):
    requests = _corpus()
    calibration, report = fit_calibration(tiny_model, requests, target="gold", device="cpu")
    assert set(calibration.temperatures) == {"choice", "boolean", "score"}
    assert all(0.1 <= value <= 4.0 for value in calibration.temperatures.values())
    for kind, after in report["nll_after"].items():
        assert after <= report["nll_before"][kind] + 1e-9


def test_calibration_round_trip(tiny_model, tmp_path):
    requests = _corpus()
    calibration, _ = fit_calibration(tiny_model, requests, target="gold", device="cpu")
    tiny_model.calibration = calibration
    directory = tmp_path / "calibrated"
    tiny_model.save(directory)
    reloaded = DecisionModel.load(directory, device="cpu")
    assert reloaded.calibration.to_dict() == calibration.to_dict()
    records = reloaded.score(requests[:2])
    assert all(record["decode_steps"] == 0 for record in records)


def test_ece_metric():
    assert _ece([1.0, 1.0], [1.0, 1.0]) == 0.0
    assert _ece([1.0, 1.0], [0.0, 0.0]) == pytest.approx(1.0)
    summary = aggregate(
        [
            {"type": "boolean", "candidate_ids": ["false", "true"], "probabilities": [0.2, 0.8],
             "gold": True, "correct": True, "decode_steps": 0},
            {"type": "boolean", "candidate_ids": ["false", "true"], "probabilities": [0.9, 0.1],
             "gold": True, "correct": False, "decode_steps": 0},
        ]
    )
    assert 0 < summary["expected_calibration_error"] <= 1
    assert summary["mean_confidence"] == pytest.approx(0.85)


def test_benchmark_modes(tiny_model):
    requests = generate_split(4, seed=4, split="dev")
    result = benchmark_model(tiny_model, requests, BenchOptions(modes=("fresh", "reuse"), repeats=1, max_latency_requests=3, device="cpu"))
    assert set(result["modes"]) == {"fresh", "reuse"}
    for mode in result["modes"].values():
        assert mode["decode_steps"] == 0
        assert mode["latency_p50_ms"] > 0
        assert mode["batch_questions"] > 0
    comparison = compare_modes(result)
    assert comparison["throughput_ratio"] > 0


def test_two_stage_logits_handles_large_candidate_sets(tiny_model):
    options = {f"c{index}": f"option {index}" for index in range(24)}
    request = Request(id="wide", state="s", questions=[make_choice_question("q", "pick one", options)])
    records = score_logits_two_stage(tiny_model.backbone, tiny_model.tokenizer, [request], should_normalize=True)
    assert len(records) == 1
    record = records[0]
    assert record["mode"] == "logits-two-stage"
    assert len(record["probabilities"]) == 24
    assert abs(sum(record["probabilities"]) - 1.0) < 1e-5
    assert record["decode_steps"] == 0
