import json
import random
import subprocess
import sys
from pathlib import Path

from minojev.maze import (
    DIRECTIONS,
    bfs_distance,
    direction_safe,
    distance,
    generate_maze,
    generate_split,
    legal_moves,
    rollout,
    state_request,
    state_object,
)
from minojev.types import teacher_distribution


def test_generated_mazes_are_solvable_and_clear():
    rng = random.Random(3)
    for _ in range(20):
        maze = generate_maze(rng)
        assert bfs_distance(maze, maze.start) is not None
        assert maze.start not in maze.walls and maze.goal not in maze.walls


def test_safety_and_distance():
    rng = random.Random(5)
    maze = generate_maze(rng)
    agent = maze.start
    assert direction_safe(maze, agent, "up") is False
    assert direction_safe(maze, agent, "left") is False
    assert legal_moves(maze, agent)
    assert distance(maze, agent) == (maze.size - 1) * 2
    assert bfs_distance(maze, maze.goal) == 0


def test_state_request_contract():
    rng = random.Random(7)
    maze = generate_maze(rng)
    request = state_request(maze, maze.start, 4)
    kinds = [(question.qid, question.kind) for question in request.questions]
    assert kinds == [
        ("move", "choice"),
        ("safe-up", "boolean"),
        ("safe-down", "boolean"),
        ("safe-left", "boolean"),
        ("safe-right", "boolean"),
        ("distance", "score"),
    ]
    state = state_object(maze, maze.start, 4)
    assert state["walls"] == [list(cell) for cell in maze.wall_list]
    for question in request.questions:
        probabilities = teacher_distribution(question, request.teacher[question.qid])
        assert probabilities is not None
        assert abs(sum(probabilities) - 1.0) < 1e-9
    move = request.teacher["move"]
    assert len(move) == 4
    safe_teacher = request.teacher["safe-up"]
    assert safe_teacher["true"] == (0.9 if request.gold["safe-up"] else 0.1)


def test_rollout_mechanics(tiny_model):
    rng = random.Random(11)
    maze = generate_maze(rng)
    result = rollout(tiny_model, maze, max_steps=5, device="cpu")
    assert result["size"] == maze.size
    assert result["optimal_steps"] is not None
    assert len(result["frames"]) <= 5
    for frame in result["frames"]:
        assert frame["move"] in DIRECTIONS
        assert abs(sum(frame["move_probabilities"].values()) - 1.0) < 1e-4
        assert 0.0 <= frame["distance_expected"] <= 3.0
    if result["frames"]:
        last = result["frames"][-1]
        reached = result["final"] == list(maze.goal)
        assert result["success"] == reached
        assert last["distance"] >= 0


def test_generate_split_is_deterministic():
    first = generate_split(4, seed=9, split="test")
    second = generate_split(4, seed=9, split="test")
    assert [request.state for request in first] == [request.state for request in second]
    assert [request.id for request in first] != [request.id for request in generate_split(4, seed=9, split="train")]


def test_maze_data_cli(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "maze.jsonl"
    result = subprocess.run(
        [sys.executable, "-m", "minojev.cli", "maze-data", "--out", str(output), "--count", "3", "--seed", "2"],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(root / "src"), "PATH": "/usr/bin:/bin"},
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 3
    assert all("questions" in row and "teacher" in row for row in rows)
