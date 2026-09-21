import json
import random
import subprocess
import sys
from pathlib import Path

from minojev.sokoban import (
    DIRECTIONS,
    apply_move,
    corner_deadlock,
    deadlocked,
    generate_sokoban,
    generate_split,
    move_legal,
    on_target,
    rollout,
    safe_move,
    solve,
    solved,
    state_object,
    state_request,
)
from minojev.types import teacher_distribution


def test_generated_levels_are_solvable():
    rng = random.Random(3)
    for _ in range(12):
        level = generate_sokoban(rng)
        assert not solved(level)
        assert len(level.boxes) == len(level.goals)
        solution = solve(level)
        assert solution is not None
        replayed = level
        for move in solution:
            assert move_legal(replayed, move)
            replayed = apply_move(replayed, move)
        assert solved(replayed)


def test_corner_deadlock_detection():
    level = generate_sokoban(random.Random(5))
    assert not deadlocked(level)
    box = sorted(level.boxes)[0]
    assert corner_deadlock(level, box) in (True, False)
    for direction in DIRECTIONS:
        moved = apply_move(level, direction) if move_legal(level, direction) else None
        if moved is not None:
            assert safe_move(level, direction) == (not deadlocked(moved))


def test_state_request_contract():
    level = generate_sokoban(random.Random(7))
    request = state_request(level, 4)
    kinds = [(question.qid, question.kind) for question in request.questions]
    assert kinds == [
        ("move", "choice"),
        ("safe-up", "boolean"),
        ("safe-down", "boolean"),
        ("safe-left", "boolean"),
        ("safe-right", "boolean"),
        ("progress", "score"),
    ]
    state = state_object(level, 4)
    lines = state.splitlines()
    assert lines[0] == f"{level.size}x{level.size}"
    map_lines = lines[1 : level.size + 1]
    assert len(map_lines) == level.size
    box_marks = sum(row.count("$") + row.count("*") for row in map_lines)
    goal_marks = sum(row.count("o") + row.count("*") + row.count("+") for row in map_lines)
    assert box_marks == len(level.boxes)
    assert goal_marks == len(level.goals)
    assert sum(row.count("@") + row.count("+") for row in map_lines) == 1
    assert any(line.startswith("player ") for line in lines)
    assert any(line.startswith("boxes ") for line in lines)
    assert any(line.startswith("goals ") for line in lines)
    for question in request.questions:
        probabilities = teacher_distribution(question, request.teacher[question.qid])
        assert probabilities is not None
        assert abs(sum(probabilities) - 1.0) < 1e-9
    assert len(request.teacher["move"]) == 4
    assert request.teacher["safe-up"]["true"] == (0.9 if request.gold["safe-up"] else 0.1)
    assert request.gold["progress"] == on_target(level)


def test_rollout_mechanics(tiny_model):
    result = rollout(tiny_model, seed=11, max_steps=5, device="cpu")
    assert result["size"] == 7
    assert isinstance(result["optimal_steps"], int)
    assert len(result["frames"]) <= 5
    for frame in result["frames"]:
        assert frame["move"] in DIRECTIONS
        assert abs(sum(frame["move_probabilities"].values()) - 1.0) < 1e-4
        assert 0 <= frame["on_target"] <= 1
    if result["frames"]:
        goals = {tuple(cell) for cell in result["goals"]}
        reached = all(tuple(box) in goals for box in result["final"]["boxes"])
        assert result["success"] == reached
        assert len(result["final"]["boxes"]) == 1


def test_generate_split_is_deterministic():
    first = generate_split(3, seed=9, split="test")
    second = generate_split(3, seed=9, split="test")
    assert [request.state for request in first] == [request.state for request in second]
    assert [request.id for request in first] != [request.id for request in generate_split(3, seed=9, split="train")]


def test_sokoban_data_cli(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "sokoban.jsonl"
    result = subprocess.run(
        [sys.executable, "-m", "minojev.cli", "sokoban-data", "--out", str(output), "--count", "3", "--seed", "2"],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(root / "src"), "PATH": "/usr/bin:/bin"},
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 3
    assert all("questions" in row and "teacher" in row for row in rows)
