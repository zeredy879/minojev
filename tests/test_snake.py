import json
import random
import subprocess
import sys
from pathlib import Path

from minojev.snake import (
    DIRECTIONS,
    advance,
    direction_safe,
    food_distance,
    generate_snake,
    generate_split,
    legal_moves,
    manhattan,
    rollout,
    state_request,
    state_object,
)
from minojev.types import teacher_distribution


def test_generated_snake_is_consistent():
    rng = random.Random(3)
    for _ in range(20):
        snake = generate_snake(rng)
        assert len(set(snake.body)) == snake.length
        assert snake.food not in snake.body
        assert snake.head == snake.body[0]
        assert food_distance(snake) is not None
        for head, tail in zip(snake.body, snake.body[1:]):
            assert manhattan(head, tail) == 1


def test_safety_and_moves():
    rng = random.Random(5)
    snake = generate_snake(rng)
    assert direction_safe(snake, "left") is False  # the neck blocks the reverse move
    for direction in DIRECTIONS:
        if direction_safe(snake, direction):
            moved, ate = advance(snake, direction)
            assert moved.head != snake.head
            assert moved.head not in moved.body[1:]
            assert moved.length == snake.length + (1 if ate else 0)


def test_state_request_contract():
    rng = random.Random(7)
    snake = generate_snake(rng)
    request = state_request(snake, 4, eaten=1)
    kinds = [(question.qid, question.kind) for question in request.questions]
    assert kinds == [
        ("move", "choice"),
        ("safe-up", "boolean"),
        ("safe-down", "boolean"),
        ("safe-left", "boolean"),
        ("safe-right", "boolean"),
        ("distance", "score"),
    ]
    state = state_object(snake, 4, 1)
    assert state["snake"] == [list(cell) for cell in snake.body]
    assert state["eaten"] == 1
    for question in request.questions:
        probabilities = teacher_distribution(question, request.teacher[question.qid])
        assert probabilities is not None
        assert abs(sum(probabilities) - 1.0) < 1e-9
    assert len(request.teacher["move"]) == 4
    assert request.teacher["safe-up"]["true"] == (0.9 if request.gold["safe-up"] else 0.1)


def test_rollout_mechanics(tiny_model):
    result = rollout(tiny_model, seed=11, max_steps=8, target_food=1, device="cpu")
    assert result["size"] == 6
    assert len(result["frames"]) <= 8
    for frame in result["frames"]:
        assert frame["move"] in DIRECTIONS
        assert abs(sum(frame["move_probabilities"].values()) - 1.0) < 1e-4
        assert frame["eaten"] >= 0
    if result["frames"]:
        assert result["success"] == (result["eaten"] >= result["target_food"])
        assert len(result["final"]) >= 3


def test_generate_split_is_deterministic():
    first = generate_split(4, seed=9, split="test")
    second = generate_split(4, seed=9, split="test")
    assert [request.state for request in first] == [request.state for request in second]
    assert [request.id for request in first] != [request.id for request in generate_split(4, seed=9, split="train")]


def test_snake_data_cli(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "snake.jsonl"
    result = subprocess.run(
        [sys.executable, "-m", "minojev.cli", "snake-data", "--out", str(output), "--count", "3", "--seed", "2"],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(root / "src"), "PATH": "/usr/bin:/bin"},
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 3
    assert all("questions" in row and "teacher" in row for row in rows)
