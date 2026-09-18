import torch

from minojev import ScoreOptions, Request, make_choice_question


def test_score_shapes_and_zero_decoding(tiny_model, sample_requests):
    records = tiny_model.score(sample_requests)
    assert len(records) == 4
    assert all(record["decode_steps"] == 0 for record in records)
    for record in records:
        assert len(record["probabilities"]) == len(record["candidate_ids"])
        assert abs(sum(record["probabilities"]) - 1.0) < 1e-5
    boolean = next(record for record in records if record["type"] == "boolean")
    assert len(boolean["probabilities"]) == 2
    score = next(record for record in records if record["type"] == "score")
    assert "expected" in score


def test_fresh_and_reuse_agree(tiny_model):
    first = Request(
        id="r1",
        state={"color": "blue", "shape": "round"},
        questions=[
            make_choice_question("a", "pick 'shape'", {"round": "round", "red": "red", "tiny": "tiny"}),
            make_choice_question("b", "pick 'color'", {"round": "round", "blue": "blue"}),
        ],
    )
    second = Request(
        id="r2",
        state={"note": "a much longer state string so the two states have different prefix lengths"},
        questions=[make_choice_question("a", "pick one", {"x": "first", "y": "second", "z": "third"})],
    )
    fresh = tiny_model.score([first, second], ScoreOptions(mode="fresh"))
    reuse = tiny_model.score([first, second], ScoreOptions(mode="reuse"))
    assert len(fresh) == len(reuse) == 3
    for left, right in zip(fresh, reuse):
        assert left["id"] == right["id"]
        assert left["candidate_ids"] == right["candidate_ids"]
        assert torch.allclose(torch.tensor(left["probabilities"]), torch.tensor(right["probabilities"]), atol=1e-5)


def test_batch_independence(tiny_model, sample_requests):
    alone = tiny_model.score([sample_requests[0]])
    together = tiny_model.score(sample_requests)
    for left, right in zip(alone, together):
        assert left["qid"] == right["qid"]
        assert torch.allclose(torch.tensor(left["probabilities"]), torch.tensor(right["probabilities"]), atol=1e-6)


def test_two_to_255_candidates(tiny_model):
    for size in (2, 17, 255):
        options = {f"c{index}": f"value {index}" for index in range(size)}
        request = Request(id=f"r{size}", state="s", questions=[make_choice_question("q", "pick one", options)])
        records = tiny_model.score([request])
        assert len(records[0]["probabilities"]) == size
        assert abs(sum(records[0]["probabilities"]) - 1.0) < 1e-4


def test_checkpoint_round_trip(tiny_model, tmp_path):
    request = Request(
        id="r",
        state="checkpoint state",
        questions=[make_choice_question("q", "pick", {"a": "first", "b": "second", "c": "third"})],
    )
    before = tiny_model.score([request])
    directory = tmp_path / "checkpoint"
    tiny_model.save(directory)
    from minojev import DecisionModel

    reloaded = DecisionModel.load(directory, device="cpu")
    after = reloaded.score([request])
    assert torch.allclose(torch.tensor(before[0]["probabilities"]), torch.tensor(after[0]["probabilities"]), atol=1e-6)
    assert (directory / "model.safetensors").exists()
    assert (directory / "head.safetensors").exists()
