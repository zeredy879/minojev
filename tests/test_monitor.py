import json
from pathlib import Path

import pytest
import torch

from minojev.monitor import MemoryBudget, TrainingMonitor, memory_snapshot, pressure_snapshot, release_memory


def test_pressure_snapshot():
    import time

    pressure_snapshot()
    time.sleep(0.2)
    snapshot = pressure_snapshot()
    assert snapshot["cpu_cores"] >= 1
    assert snapshot["process_cpu_percent"] is None or snapshot["process_cpu_percent"] >= 0
    assert snapshot["system_memory_percent"] is None or 0 <= snapshot["system_memory_percent"] <= 100
    assert snapshot["load1_per_core"] is None or snapshot["load1_per_core"] >= 0


def test_memory_snapshot_shape():
    snapshot = memory_snapshot()
    assert "cpu_peak_rss_gb" in snapshot
    assert snapshot["cpu_peak_rss_gb"] > 0
    assert "mps_allocated_gb" in snapshot and "mps_driver_gb" in snapshot


def test_memory_budget_guard():
    with pytest.raises(MemoryError):
        MemoryBudget(max_gb=0.000001).check("unit test")
    snapshot = MemoryBudget(max_gb=None).check("unit test")
    assert snapshot["cpu_peak_rss_gb"] > 0


def test_training_monitor_writes_status_and_metrics(tmp_path):
    monitor = TrainingMonitor(tmp_path / "run", total_steps=10, log_every=1, label="unit", stream=open("/dev/null", "w"))
    for step in range(1, 4):
        monitor.log(step, "phase", loss=0.5, lr=1e-3, grad_norm=1.2, tokens=step * 100)
    monitor.note("hello")
    status = json.loads((tmp_path / "run" / "status.json").read_text())
    assert status["note"] == "hello"
    lines = (tmp_path / "run" / "metrics.jsonl").read_text().strip().splitlines()
    assert len(lines) == 4
    entry = json.loads(lines[0])
    assert entry["step"] == 1 and entry["loss"] == 0.5 and "memory" in entry


def test_watch_server_endpoints(tmp_path):
    import json as jsonlib
    import threading
    import urllib.request

    from minojev.watch import build_watch_server

    run = tmp_path / "run"
    run.mkdir()
    (run / "status.json").write_text(jsonlib.dumps({"label": "demo", "step": 3, "total_steps": 10, "memory": {"mps_driver_gb": 1.5}}))
    (run / "metrics.jsonl").write_text(
        "\n".join(jsonlib.dumps({"step": index, "loss": 1.0 - index / 10}) for index in range(1, 4)) + "\n"
    )
    (run / "train_log.json").write_text(jsonlib.dumps([{"step": 2, "dev": {"accuracy": 0.5, "teacher_ce": 1.2}}]))
    server = build_watch_server(run, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        base = f"http://127.0.0.1:{port}"
        status = jsonlib.loads(urllib.request.urlopen(f"{base}/api/status", timeout=30).read())
        assert status["label"] == "demo"
        metrics = jsonlib.loads(urllib.request.urlopen(f"{base}/api/metrics?limit=2", timeout=30).read())
        assert [entry["step"] for entry in metrics] == [2, 3]
        log = jsonlib.loads(urllib.request.urlopen(f"{base}/api/train_log", timeout=30).read())
        assert log[0]["dev"]["accuracy"] == 0.5
        page = urllib.request.urlopen(f"{base}/", timeout=30).read().decode()
        assert "training monitor" in page
    finally:
        server.shutdown()
        server.server_close()


def test_chunked_calibration_matches_single_shot(tmp_path):
    from minojev.domains import generate_split
    from minojev.posttrain import PostTrainConfig, build_model
    from minojev.calibrate import fit_calibration

    class TinyConfig(PostTrainConfig):
        pass

    config = PostTrainConfig(backbone="tiny", mode="head", device="cpu")
    # Build a tiny offline HF backbone instead of downloading one.
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from minojev import ByteTokenizer, DecisionModel
    from minojev.backbone import HFBackbone
    from minojev.heads import DecisionHead

    torch.manual_seed(0)
    hf = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=259,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=128,
            max_position_embeddings=512,
        )
    )
    tokenizer = ByteTokenizer()
    model = DecisionModel(HFBackbone(hf, tokenizer=tokenizer, source="tiny"), tokenizer, DecisionHead(64, attention_dim=32, num_heads=4))
    requests = generate_split(6, seed=13, split="dev", domain="support")
    single, _ = fit_calibration(model, requests, target="gold", device="cpu", batch_requests=1000)
    chunked, _ = fit_calibration(model, requests, target="gold", device="cpu", batch_requests=2)
    for kind in ("choice", "boolean", "score"):
        assert single.temperature(kind) == pytest.approx(chunked.temperature(kind), rel=1e-6)
    release_memory()
