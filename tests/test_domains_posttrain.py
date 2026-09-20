import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from minojev import ByteTokenizer, DecisionModel
from minojev.backbone import HFBackbone
from minojev.domains import DOMAIN_NAMES, generate_split
from minojev.heads import DecisionHead
from minojev.model import ScoreOptions
from minojev.posttrain import PostTrainConfig, posttrain, train_head_cached
from minojev.types import teacher_distribution


def _tiny_hf_model(vocab_size=259):
    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=vocab_size,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        max_position_embeddings=512,
    )
    return Qwen2ForCausalLM(config).eval()


def _tiny_hf_decision_model(directory, vocab_size=259):
    from minojev.backbone import load_hf_backbone
    from minojev.tokenizer import save_tokenizer

    base = Path(directory) / "base"
    _tiny_hf_model(vocab_size).save_pretrained(base)
    save_tokenizer(ByteTokenizer(), base)
    backbone = load_hf_backbone(str(base), dtype="float32", device="cpu")
    return DecisionModel(backbone, backbone.tokenizer, DecisionHead(64, attention_dim=32, num_heads=4))


def test_domain_generation_contract():
    for domain in DOMAIN_NAMES:
        requests = generate_split(3, seed=5, split="test", domain=domain)
        assert len(requests) == 3
        for request in requests:
            assert len(request.questions) == 3
            assert sorted(question.kind for question in request.questions) == ["boolean", "choice", "score"]
            for question in request.questions:
                probabilities = teacher_distribution(question, request.teacher[question.qid])
                assert probabilities is not None
                assert abs(sum(probabilities) - 1.0) < 1e-9
            assert request.state
    first = generate_split(2, seed=7, split="dev", domain="refunds")
    second = generate_split(2, seed=7, split="dev", domain="refunds")
    assert [request.state for request in first] == [request.state for request in second]


def test_hf_backbone_save_load_roundtrip(tmp_path):
    model = _tiny_hf_decision_model(tmp_path)
    requests = generate_split(3, seed=11, split="dev", domain="support")
    before = model.score(requests, ScoreOptions(device="cpu"))
    model.save(tmp_path / "checkpoint")
    loaded = DecisionModel.load(tmp_path / "checkpoint", device="cpu")
    after = loaded.score(requests, ScoreOptions(device="cpu"))
    assert len(before) == len(after) == 9
    for left, right in zip(before, after):
        assert left["candidate_ids"] == right["candidate_ids"]
        assert torch.allclose(torch.tensor(left["probabilities"]), torch.tensor(right["probabilities"]), atol=1e-6)


def test_head_posttrain_smoke(tmp_path):
    model = _tiny_hf_decision_model(tmp_path)
    requests = generate_split(4, seed=3, split="dev", domain="leads")
    config = PostTrainConfig(mode="head", steps=4, eval_every=2, batch_requests=2, cache_batch_requests=2, device="cpu")
    summary = train_head_cached(model, requests, requests[:2], config, tmp_path / "head")
    assert summary["best_step"] is not None
    assert (tmp_path / "head" / "checkpoint" / "head.safetensors").exists()
    assert not (tmp_path / "head" / "checkpoint" / "backbone").exists()
    log = json.loads((tmp_path / "head" / "train_log.json").read_text())
    assert len(log) == config.steps
    assert all(isinstance(entry, dict) for entry in log)
    assert sum(1 for entry in log if "dev" in entry) == 2
    loaded = DecisionModel.load(tmp_path / "head" / "checkpoint", device="cpu")
    records = loaded.score(requests[:1], ScoreOptions(device="cpu"))
    assert len(records) == 3


def test_lora_posttrain_smoke(tmp_path):
    model = _tiny_hf_decision_model(tmp_path)
    requests = generate_split(4, seed=4, split="dev", domain="moderation")
    config = PostTrainConfig(
        mode="lora",
        steps=2,
        head_steps=1,
        eval_every=1,
        batch_requests=2,
        device="cpu",
        save_dtype="float32",
    )
    summary = posttrain(model, requests, requests[:2], config, tmp_path / "lora")
    assert summary["best_step"] is not None
    merged = DecisionModel.load(tmp_path / "lora" / "checkpoint", device="cpu")
    records = merged.score(requests[:1], ScoreOptions(device="cpu"))
    assert len(records) == 3
    assert (tmp_path / "lora" / "checkpoint" / "backbone" / "config.json").exists()


def test_serve_http_endpoint(tmp_path):
    import json as jsonlib
    import threading
    import urllib.request

    model = _tiny_hf_decision_model(tmp_path)
    checkpoint = tmp_path / "serve-ckpt"
    model.save(checkpoint)

    from minojev.serve import build_server

    server = build_server(str(checkpoint), port=0, device="cpu")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        payload = jsonlib.dumps(
            {"id": "live-1", "state": "state", "question": "pick one", "options": {"a": "alpha", "b": "beta"}}
        ).encode()
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/score", data=payload, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            body = jsonlib.loads(response.read())
        assert body["count"] == 1
        record = body["records"][0]
        assert record["decode_steps"] == 0
        assert abs(sum(record["probabilities"]) - 1.0) < 1e-4
    finally:
        server.shutdown()
        server.server_close()


def test_domain_data_cli(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "domains.jsonl"
    result = subprocess.run(
        [sys.executable, "-m", "minojev.cli", "domain-data", "--out", str(output), "--count", "6", "--seed", "2"],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 6
    assert all("questions" in row and "teacher" in row for row in rows)
