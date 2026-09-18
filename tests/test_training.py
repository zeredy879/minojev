import pytest
import torch

from minojev import DecisionModel
from minojev.train import TrainConfig, batch_loss, train
from minojev.synth import generate_split


def _teacher_ce(records):
    terms = [
        -sum(
            record["teacher"][candidate_id] * torch.log(torch.tensor(max(probability, 1e-9))).item()
            for candidate_id, probability in zip(record["candidate_ids"], record["probabilities"])
        )
        for record in records
    ]
    return sum(terms) / len(terms)


@pytest.mark.slow
def test_training_converges_on_fixed_requests(tmp_path):
    requests = generate_split(6, seed=11, split="train")
    model = DecisionModel.new(hidden_size=64, num_layers=2, num_heads=4, intermediate_size=128)
    before = model.score(requests)
    config = TrainConfig(
        steps=100,
        head_steps=10,
        batch_requests=6,
        eval_every=25,
        learning_rate=5e-4,
        head_lr=1e-3,
        objective="teacher_ce",
        seed=11,
        device="cpu",
    )
    summary = train(model, requests, requests, config, tmp_path / "run")
    assert summary["best_step"] is not None
    loaded = DecisionModel.load(tmp_path / "run" / "checkpoint", device="cpu")
    after = loaded.score(requests)
    assert _teacher_ce(after) < _teacher_ce(before) * 0.75
    assert sum(bool(record.get("correct")) for record in after) / len(after) > 0.8
    assert (tmp_path / "run" / "train_log.json").exists()


def test_batch_loss_backpropagates(tiny_model):
    requests = generate_split(2, seed=2, split="train")
    encoded = tiny_model.encode(requests)
    loss = batch_loss(tiny_model, encoded, "teacher_ce")
    loss.backward()
    gradients = [parameter.grad for parameter in tiny_model.head.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
