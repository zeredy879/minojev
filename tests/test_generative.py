import json
from pathlib import Path

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from minojev import ByteTokenizer, DecisionModel
from minojev.backbone import HFBackbone, load_hf_backbone
from minojev.compare import CompareOptions, compare_workload, render_markdown, write_report
from minojev.generative import generative_prompt, parse_answer, run_generative_baseline, summarize_generative
from minojev.heads import DecisionHead
from minojev.tokenizer import save_tokenizer
from minojev.types import Request, make_boolean_question, make_choice_question


def _tiny_lm():
    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=259,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        max_position_embeddings=512,
    )
    return Qwen2ForCausalLM(config).eval()


def _tiny_decision_model(tmp_path):
    base = tmp_path / "base"
    _tiny_lm().save_pretrained(base)
    save_tokenizer(ByteTokenizer(), base)
    backbone = load_hf_backbone(str(base), dtype="float32", device="cpu")
    return DecisionModel(backbone, backbone.tokenizer, DecisionHead(64, attention_dim=32, num_heads=4))


def _requests():
    return [
        Request(
            id="r1",
            state={"text": "How do I add a card?"},
            questions=[make_choice_question("q", "Which category applies?", {"card_linking": "card_linking", "atm": "atm", "transfer": "transfer"})],
            gold={"q": "card_linking"},
            teacher={"q": {"card_linking": 0.85, "atm": 0.075, "transfer": 0.075}},
        ),
        Request(
            id="r2",
            state={"content": "Great product, works well."},
            questions=[make_boolean_question("q", "Is this text positive?")],
            gold={"q": True},
            teacher={"q": {"true": 0.9, "false": 0.1}},
        ),
    ]


def test_select_balanced_covers_sources():
    from minojev.compare import select_balanced

    requests = []
    for family in ("a", "b", "c"):
        for index in range(10):
            question = make_boolean_question("q", "ok?")
            question.family = family
            requests.append(Request(id=f"{family}-{index}", state="s", questions=[question]))
    selected = select_balanced(requests, 9)
    assert len(selected) == 9
    assert {request.questions[0].family for request in selected} == {"a", "b", "c"}
    assert len({request.id for request in selected}) == 9


def test_parse_answer_variants():
    question = make_choice_question("q", "pick", {"alpha": "alpha", "beta": "beta", "gamma": "gamma"})
    assert parse_answer("B", question) == ("beta", "letter")
    assert parse_answer("The answer is C.", question) == ("gamma", "letter")
    assert parse_answer("alpha", question)[0] == "alpha"
    assert parse_answer("", question) == (None, None)
    boolean = make_boolean_question("b", "is it?")
    assert parse_answer("true", boolean) == ("true", "boolean")
    assert parse_answer("No.", boolean) == ("false", "boolean")


def test_generative_prompt_lists_options():
    request = _requests()[0]
    prompt = generative_prompt(request, request.questions[0])
    assert "A. card_linking" in prompt
    assert "C. transfer" in prompt
    assert "option letter" in prompt


def test_generative_baseline_tiny_model(tmp_path):
    model = _tiny_lm()
    records = run_generative_baseline(model, ByteTokenizer(), _requests(), None)
    assert len(records) == 2
    for record in records:
        assert record["output_tokens"] >= 1
        assert record["total_seconds"] > 0
        assert record["first_token_seconds"] > 0
        assert "parsed" in record
    summary = summarize_generative(records)
    assert summary["decisions"] == 2
    assert 0 <= summary["parse_failure_rate"] <= 1
    assert summary["output_tokens_per_decision"] >= 1
    assert summary["latency_p50_ms"] is not None


def test_compare_workload_report(tmp_path):
    model = _tiny_decision_model(tmp_path)
    generative = _tiny_lm()
    report = compare_workload(model, _requests(), generative, ByteTokenizer(), CompareOptions(device="cpu", max_new_tokens=3))
    assert report["requests"] == 2
    assert report["decisions"] == 2
    assert report["engine"]["output_tokens_per_decision"] == 0.0
    assert report["generative"]["output_tokens_per_decision"] >= 1
    assert "latency_p50_ms" in report["engine"]
    assert set(report["per_source"]) <= {"card_linking", "unknown", ""} or report["per_source"]
    markdown = render_markdown(report)
    assert "Decision engine vs generative baseline" in markdown
    assert "output tokens / decision" in markdown
    write_report(report, tmp_path / "report.json", tmp_path / "report.md")
    written = json.loads((tmp_path / "report.json").read_text())
    assert written["engine"]["output_tokens_per_decision"] == 0.0
    assert "generative baseline" in (tmp_path / "report.md").read_text()
